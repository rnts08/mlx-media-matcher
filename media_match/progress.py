"""Live progress for every long stage: how fast, how long, and what is stuck.

A scan of 20k files is minutes of unexplained silence. The old output printed
``preparing [4000/20000]`` twenty times, which answers "is it alive" but not
"is it fast" or "is it stuck" — and it never said which files were in flight,
so a single pathological image looked exactly like slow-but-progressing work.
This draws one live status instead, showing the things that actually tell you
whether to wait or to interrupt:

    ⠹ indexing  14,208/20,000  [█████████████░░░░░░░░]  71%  186/s↗  eta 0:50
    now  Photos/IMG_4471.jpg, Photos/IMG_4472.jpg   ring  1.0GB/1.0GB
    cpu 38.2s   gpu 11.4s (30%)   read 1.2GB

The rate and ETA are not a lifetime average. They are extrapolated from the
last 60 seconds of completed items, the same way an OS copy dialog estimates
"about 40 seconds remaining": a stage that starts fast and then hits a folder
of huge videos slows down instead of stubbornly promising the original time.
The arrow beside the rate says whether it is speeding up (↗) or slowing down
(↘) compared with the previous frame.

The three pipeline numbers to actually read:

* **ring** — how full the read buffer is. Full means the disk is ahead of the
  CPU and adding ``--io-workers`` will not help; empty means the readers are
  the bottleneck and it will.
* **cpu** — CPU-seconds burned, summed over all threads. Rising faster than
  wall time means the workers are genuinely parallel; flat means they are
  fighting over the GIL and more threads will not help.
* **gpu** — time inside embedding calls. The percentage is GPU utilisation, so
  a low number with a high ring is exactly the "GPU is idle" case the
  ring-buffer work was meant to fix.

Rendering happens on its own thread at a fixed rate. The pipeline only ever
mutates a small state object under a lock, so instrumentation costs
microseconds and never sits in front of a decode.

The describe stage reuses the same bar and the same counters. Its buffer is the
prefetch window rather than a byte ring — ``window 6/8`` frames decoded ahead of
the caption model — so "full" means the GPU is the bottleneck and "empty" means
it is starving.

Everything degrades quietly: no TTY (pipes, CI, tests) means nothing is drawn
at all and the caller falls back to its own periodic lines.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

# Redraws per second. 10 Hz is fast enough to look continuous and slow enough
# that rendering never competes with decoding for the GIL.
FRAME_HZ = 10

# How many filenames to show; more than this and the line wraps.
MAX_NAMES = 2

# Filenames longer than this are truncated from the left, so the interesting
# part (the name) survives even in a deep tree.
MAX_NAME_CHARS = 34

# The ETA is extrapolated from this trailing window, like a copy dialog. A
# lifetime average is misleading: it keeps promising the old speed after a
# tree of video files has slowed the pipeline down.
SAMPLE_WINDOW = 60.0

# Do not divide by a window shorter than this; the first seconds produce
# absurd rates and a nonsense ETA.
MIN_RATE_SPAN = 1.0

# Record at most one sample per this interval. At 500 items/s an unthrottled
# sampler would keep ~30k tuples for one minute; this keeps a few hundred.
SAMPLE_INTERVAL = 0.1

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class PipelineState:
    """Everything the renderer needs, mutated by the pipeline's threads."""

    total: int = 0
    done: int = 0
    #: paths currently being read from disk (bounded)
    reading: list[str] = field(default_factory=list)
    #: paths currently being decoded/hashed/preprocessed (bounded)
    decoding: list[str] = field(default_factory=list)
    #: (used, capacity) of the pipeline's buffer: the read ring during a scan,
    #: the prefetch window during describe. Bytes for the ring, a plain count
    #: for the window, so the unit is carried by ring_bytes.
    ring_used: int = 0
    ring_cap: int = 0
    ring_label: str = "ring"
    ring_bytes: bool = True
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    bytes_read: int = 0
    started: float = 0.0
    finished: float = 0.0
    #: set by the pipeline when an item failed
    errors: int = 0
    #: what the GPU stage is doing when no file is in flight ("embedding")
    phase: str = ""
    #: trailing-window rate (items/s) and seconds remaining; 0/-1 means the
    #: renderer should fall back to the elapsed average while warming up.
    sampled_rate: float = 0.0
    eta: float = -1.0
    #: +1 speeding up, -1 slowing down, 0 steady/unknown
    trend: int = 0
    #: what to draw before the label: a spinner while running, ✓/! when done
    spinner: str = ""
    #: an optional one-line note shown after the phase ("model ready")
    note: str = ""


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if abs(n) < 1024.0:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    # Keep a decimal at GB/TB: rounding 1.5GB to "2GB" overstates a full
    # buffer by a third, which is exactly the number people tune on.
    return f"{n:.1f}GB" if n < 1024 else f"{n / 1024:.1f}TB"


def human_time(seconds: float) -> str:
    """0.4s / 12.0s / 3.4m / 2h05m — never a bare '0.0s'.

    Sub-minute values keep one decimal: this is a live counter, and dropping
    the fraction makes a fast-changing number look frozen between updates.
    """
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def sampled_rate(samples, now: float,
                 window: float = SAMPLE_WINDOW) -> float:
    """Items per second over the trailing ``window``, or 0 while warming up.

    ``samples`` is an iterable of ``(monotonic_time, completed_count)`` pairs.
    The sample just before the cutoff is kept as the baseline, so a 60-second
    window covers 60 seconds rather than whichever two samples happen to sit
    inside it. This is the copy-dialog behaviour: recent speed, not a lifetime
    average.
    """
    points: list[tuple[float, int]] = []
    baseline: tuple[float, int] | None = None
    cutoff = now - window
    for stamp, done in samples:
        if stamp <= cutoff:
            baseline = (stamp, done)
        else:
            points.append((stamp, done))
    if baseline is not None:
        points.insert(0, baseline)
    if len(points) < 2:
        return 0.0
    first_t, first_done = points[0]
    last_t, last_done = points[-1]
    span = last_t - first_t
    if span < MIN_RATE_SPAN:
        return 0.0
    return max(0.0, (last_done - first_done) / span)


def eta_seconds(rate: float, total: int, done: int) -> float | None:
    """Seconds left at ``rate``, or None when there is nothing to estimate."""
    if rate <= 0 or total <= done:
        return None
    return (total - done) / rate


def short_name(path: str, root: str, limit: int = MAX_NAME_CHARS) -> str:
    """Path relative to the root, middle-truncated so the filename survives."""
    try:
        shown = os.path.relpath(path, root)
    except ValueError:  # pragma: no cover - different drive on Windows
        shown = path
    if len(shown) <= limit:
        return shown
    keep = limit - 1
    return "…" + shown[-keep:]


def bar(fraction: float, width: int) -> str:
    """A fixed-width block bar; clamps out-of-range fractions."""
    fraction = 0.0 if fraction < 0 else (1.0 if fraction > 1 else fraction)
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def render(state: PipelineState, root: str, label: str, *,
           width: int = 80, spinner: str | None = None) -> str:
    """The full status block as a single string (no trailing newline).

    ``spinner`` lets the live renderer animate without mutating the state it
    is about to read; direct callers (tests, non-TTY summaries) get the state's
    own glyph or a neutral one.
    """
    now = time.perf_counter()
    elapsed = (state.finished or now) - state.started
    total = state.total
    done = state.done
    fraction = (done / total) if total else 0.0

    # Prefer the trailing-window rate recorded by ProgressBar. Fall back to
    # the elapsed average for direct callers, but not before the window is
    # long enough to be meaningful: the first frame must not print "648,493/s"
    # or an ETA of zero.
    rate = state.sampled_rate
    if rate <= 0 and elapsed >= MIN_RATE_SPAN:
        rate = done / elapsed
    remaining = eta_seconds(rate, total, done)
    if state.eta >= 0:
        remaining = state.eta

    glyph = spinner if spinner is not None else state.spinner
    if state.finished:
        glyph = "✓" if not state.errors else "!"
    glyph = glyph or "…"

    head = f"{glyph} {label:<9} {done:>6,}/{total:<6,}"
    if width >= 60:
        head += f" [{bar(fraction, 20)}] {fraction * 100:3.0f}%"
    if rate > 0:
        # Scans run at hundreds per second, describing at about one. Whole
        # numbers turned a working describe stage into "0/s" for its entire
        # run, which reads as stuck rather than slow.
        rate_text = f"{rate:,.1f}/s" if rate < 10 else f"{rate:,.0f}/s"
        trend = "↗" if state.trend > 0 else ("↘" if state.trend < 0 else "")
        head += f"  {rate_text}{trend}"
    if remaining is not None and total and not state.finished:
        head += f"  eta {human_time(remaining)}"

    # Line 2: which files are in flight right now.
    names = state.decoding or state.reading
    detail = "  ".join(short_name(p, root) for p in names[:MAX_NAMES])
    if len(names) > MAX_NAMES:
        detail += f"  (+{len(names) - MAX_NAMES} more)"
    if not detail:
        # A GPU batch in flight has no reader/decoder to name; saying "idle"
        # while the GPU is busy reads as a stalled scan. A finished frame has
        # nothing left in flight, so say "complete" instead.
        detail = (state.phase
                  or ("complete" if state.finished else "idle"))
    if state.note:
        detail += f"  {state.note}"

    ring = ""
    if state.ring_cap:
        used, cap = state.ring_used, state.ring_cap
        share = used / cap if cap else 0.0
        if state.ring_bytes:
            shown = f"{human_bytes(used)}/{human_bytes(cap)}"
        else:
            shown = f"{used}/{cap}"
        ring = f"{state.ring_label} {shown} {share * 100:3.0f}%"

    # GPU time is wall time spent inside forward passes, so it cannot exceed
    # the elapsed window. Clamp anyway: the first frame is drawn microseconds
    # after start(), and dividing by that produced percentages in the millions.
    gpu_pct = 0.0
    if elapsed > 0.001:
        gpu_pct = min(100.0, state.gpu_seconds / elapsed * 100)
    times = f"elapsed {human_time(elapsed)}   cpu {human_time(state.cpu_seconds)}   gpu " \
            f"{human_time(state.gpu_seconds)} ({gpu_pct:.0f}%)"
    if state.bytes_read:
        times += f"   read {human_bytes(state.bytes_read)}"
    if state.errors:
        times += f"   errors {state.errors}"

    second = "  ".join(x for x in (detail, ring, times) if x)
    if width and len(head) > width:
        head = head[: width - 1] + "…"
    if width and len(second) > width:
        second = second[: width - 1] + "…"
    return f"{head}\n{second}"


class ProgressBar:
    """A live, thread-safe progress display.

    Construct with ``enabled`` decided up front; when disabled every method is
    a no-op, so the pipeline can call them unconditionally without a TTY check
    at each call site.
    """

    def __init__(self, label: str, root: str, total: int = 0, *,
                 stream=None, enabled: bool | None = None,
                 frame_hz: int = FRAME_HZ):
        self.label = label
        self.root = root
        self.state = PipelineState(total=total)
        self._stream = stream if stream is not None else sys.stderr
        if enabled is None:
            enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self.enabled = enabled
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._drawn_lines = 0
        self._frame_hz = max(1, frame_hz)
        self._stopped = False
        self._frame = 0
        self._last_sample_at = 0.0
        self._last_rate = 0.0
        # (monotonic time, completed count) for the trailing-window ETA.
        self._samples: deque[tuple[float, int]] = deque()

    # -- pipeline-facing API (cheap; called from worker threads) ---------------

    def note_reading(self, path: str) -> None:
        with self._lock:
            self.state.reading.append(path)
            self._trim_locked()

    def note_decoding(self, path: str) -> None:
        with self._lock:
            self.state.decoding.append(path)
            self._trim_locked()

    def in_flight(self) -> list[str]:
        """Names of the files being read or decoded right now (a copy)."""
        with self._lock:
            return list(self.state.decoding or self.state.reading)

    def phase(self) -> str:
        with self._lock:
            return self.state.phase

    def note_phase(self, text: str) -> None:
        with self._lock:
            self.state.phase = text

    def note_finished(self, path: str) -> None:
        with self._lock:
            if path in self.state.decoding:
                self.state.decoding.remove(path)
            elif path in self.state.reading:
                self.state.reading.remove(path)

    def note_completed(self, count: int = 1) -> None:
        """Record ``count`` finished items (one by default)."""
        with self._lock:
            self.state.done += count
            self._sample_locked(time.perf_counter())

    def set_total(self, total: int) -> None:
        """Set or correct the denominator, e.g. after planning a stage."""
        with self._lock:
            total = max(0, int(total))
            changed = total != self.state.total
            self.state.total = total
            if changed:
                self._sample_locked(time.perf_counter())

    def set_done(self, done: int) -> None:
        """Set the absolute completion count (for batch-oriented stages)."""
        with self._lock:
            self.state.done = max(0, done)
            self._sample_locked(time.perf_counter())

    def _sample_locked(self, now: float) -> None:
        """Keep a throttled history for the trailing-window rate/ETA."""
        if (not self._samples
                or now - self._last_sample_at >= SAMPLE_INTERVAL
                or self.state.done >= self.state.total > 0):
            self._samples.append((now, self.state.done))
            self._last_sample_at = now
        # Keep a little more than the window so a sample just outside the
        # cutoff is still available as the baseline.
        cutoff = now - (SAMPLE_WINDOW * 2)
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _rate_locked(self, now: float) -> float:
        return sampled_rate(self._samples, now)

    def note_error(self) -> None:
        with self._lock:
            self.state.errors += 1

    def set_ring(self, used: int, cap: int) -> None:
        with self._lock:
            self.state.ring_used = used
            self.state.ring_cap = cap

    def set_window(self, used: int, cap: int, label: str = "window") -> None:
        """Report a buffer measured in items rather than bytes.

        The describe stage's analogue of the read ring is its prefetch window:
        how many frames the CPU has decoded ahead of the VLM. Full means the
        GPU is the bottleneck, empty means it is starving.
        """
        with self._lock:
            self.state.ring_used = used
            self.state.ring_cap = cap
            self.state.ring_label = label
            self.state.ring_bytes = False

    def note_read(self, nbytes: int) -> None:
        with self._lock:
            self.state.bytes_read += nbytes

    def note_cpu(self, seconds: float) -> None:
        with self._lock:
            self.state.cpu_seconds = seconds

    def note_gpu(self, seconds: float) -> None:
        with self._lock:
            self.state.gpu_seconds += seconds

    def _trim_locked(self) -> None:
        # Keep the lists bounded: a path that is still "in flight" after a
        # worker died would otherwise stay on screen forever, and an unbounded
        # list would grow for the whole run.
        if len(self.state.decoding) > MAX_NAMES + 1:
            del self.state.decoding[:-MAX_NAMES]
        if len(self.state.reading) > MAX_NAMES + 1:
            del self.state.reading[:-MAX_NAMES]

    # -- rendering ------------------------------------------------------------

    def start(self) -> "ProgressBar":
        if not self.enabled:
            return self
        self.state.started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="progress",
                                        daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        interval = 1.0 / self._frame_hz
        while not self._stop.wait(interval):
            self.draw()

    def draw(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            state = self.state
            now = time.perf_counter()
            previous = self._last_rate
            rate = self._rate_locked(now)
            if rate > 0 and previous > 0:
                # Compare against the previous frame's trailing rate. A 5%
                # deadband keeps a noisy sample from flickering the arrow.
                if rate > previous * 1.05:
                    state.trend = 1
                elif rate < previous * 0.95:
                    state.trend = -1
                else:
                    state.trend = 0
            else:
                state.trend = 0
            if rate > 0:
                self._last_rate = rate
            state.sampled_rate = rate
            state.eta = eta_seconds(rate, state.total, state.done) or -1.0
            self._frame += 1
            spinner = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        width = _term_width(self._stream)
        text = render(state, self.root, self.label, width=width,
                      spinner=spinner)
        lines = [_fit(line, width) for line in text.split("\n")]
        # Redraw the whole block in place.
        #
        # The previous renderer joined its two lines with a bare \r and ended
        # every frame with clear-to-end-of-line. That erased the block before
        # anyone could read it: in a real terminal the scan showed no progress
        # at all, even though draw() ran ten times a second. Move up to the
        # first line of the previous block, then erase and rewrite each line,
        # leaving the cursor just below it for the next frame.
        chunks: list[str] = []
        if self._drawn_lines:
            chunks.append(f"\033[{self._drawn_lines}F")
        for line in lines:
            chunks.append("\r\033[2K" + line + "\n")
        try:
            self._stream.write("".join(chunks))
            self._stream.flush()
            self._drawn_lines = len(lines)
        except (OSError, ValueError, UnicodeEncodeError):
            # A terminal that cannot encode the block/spinner glyphs falls
            # back to the caller's periodic status lines.
            self.enabled = False

    def stop(self) -> None:
        """Finish, leaving the final state on screen with a newline."""
        # Several call sites own the same bar (a stage helper and the finally
        # block around it). Stop is idempotent so the final frame cannot be
        # printed twice.
        if self._stopped:
            return
        self._stopped = True
        # Stop the renderer even if it disabled itself after a failed write:
        # otherwise a live thread would keep waking for the rest of the run.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.enabled:
            return
        with self._lock:
            self.state.finished = time.perf_counter()
            self.state.eta = -1.0
        # draw() leaves the cursor below the (newline-terminated) block.
        self.draw()
        self.enabled = False

    # Context manager so an exception cannot leave the cursor mid-line.
    def __enter__(self) -> "ProgressBar":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class Spinner:
    """A one-line animated wait indicator for model loads and short operations.

    It exists so a multi-second weight download is visibly alive without
    fighting a full progress bar. On a non-terminal it writes nothing: callers
    that need log lines (``console.load_model``) emit plain text instead.
    """

    def __init__(self, label: str, *, stream=None,
                 enabled: bool | None = None, frame_hz: int = FRAME_HZ):
        self.label = label
        self._stream = stream if stream is not None else sys.stderr
        if enabled is None:
            enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self.enabled = enabled
        self._frame_hz = max(1, frame_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._drawn = False

    def start(self) -> "Spinner":
        if not self.enabled:
            return self
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="spinner",
                                        daemon=True)
        self._thread.start()
        self.draw()
        return self

    def _run(self) -> None:
        while not self._stop.wait(1.0 / self._frame_hz):
            self.draw()

    def draw(self) -> None:
        if not self.enabled:
            return
        glyph = SPINNER_FRAMES[int(
            (time.perf_counter() - self._started) * self._frame_hz
        ) % len(SPINNER_FRAMES)]
        elapsed = time.perf_counter() - self._started
        self._write(f"\r\033[2K{glyph} {self.label}  {elapsed:4.1f}s")

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
            self._drawn = True
        except UnicodeEncodeError:
            try:
                self._stream.write(text.encode("ascii", "replace").decode("ascii"))
                self._stream.flush()
            except (OSError, ValueError, UnicodeError):
                self.enabled = False
        except (OSError, ValueError):  # pragma: no cover - closed terminal
            self.enabled = False

    def stop(self, final: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if not self.enabled:
            return
        if final is None:
            final = f"✓ {self.label} done"
        # Replace the animated line with a stable final line.
        self._write(f"\r\033[2K{final}\n")
        self.enabled = False

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _fit(text: str, width: int) -> str:
    """Keep one rendered line on one terminal row.

    A wrapped line would break the cursor-up arithmetic on the next frame, so
    long detail lines are cut to the usable width instead of wrapping.
    """
    if width and len(text) > width:
        return text[: max(0, width - 1)] + "…"
    return text


def _term_width(stream) -> int:
    """Usable width, defaulting to 80 when the terminal size is unknown.

    get_terminal_size raises, or reports 0 columns, for a pty that has not been
    sized yet. Falling back to 40 there silently dropped the bar from the
    first line, which made the display look broken rather than narrow.
    """
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (OSError, ValueError, AttributeError):
        return 80
    if not columns or columns <= 0:
        return 80
    return max(40, columns - 1)