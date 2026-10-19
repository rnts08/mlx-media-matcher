"""Live progress for a scan: what is happening, and whether it is healthy.

A scan of 20k files is minutes of unexplained silence. The old output printed
``preparing [4000/20000]`` twenty times, which answers "is it alive" but not
"is it fast" or "is it stuck" — and it never said which files were in flight,
so a single pathological image looked exactly like slow-but-progressing work.

This draws one live status instead, showing the things that actually tell you
whether to wait or to interrupt:

    indexing  14,208/20,000  [█████████████░░░░░░░░]  71%  186/s  eta 0:50
    now  Photos/IMG_4471.jpg, Photos/IMG_4472.jpg   ring  1.0GB/1.0GB
    cpu 38.2s   gpu 11.4s (30%)   read 1.2GB

The three numbers to actually read:

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

Everything degrades quietly: no TTY (pipes, CI, tests) means nothing is drawn
at all and the caller falls back to its own periodic lines.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field

# Redraws per second. 10 Hz is fast enough to look continuous and slow enough
# that rendering never competes with decoding for the GIL.
FRAME_HZ = 10

# How many filenames to show; more than this and the line wraps.
MAX_NAMES = 2

# Filenames longer than this are truncated from the left, so the interesting
# part (the name) survives even in a deep tree.
MAX_NAME_CHARS = 34


@dataclass
class PipelineState:
    """Everything the renderer needs, mutated by the pipeline's threads."""

    total: int = 0
    done: int = 0
    #: paths currently being read from disk (bounded)
    reading: list[str] = field(default_factory=list)
    #: paths currently being decoded/hashed/preprocessed (bounded)
    decoding: list[str] = field(default_factory=list)
    #: (used bytes, capacity) of the read ring
    ring_used: int = 0
    ring_cap: int = 0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    bytes_read: int = 0
    started: float = 0.0
    finished: float = 0.0
    #: set by the pipeline when an item failed
    errors: int = 0


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
           width: int = 80) -> str:
    """The full status block as a single string (no trailing newline)."""
    now = time.perf_counter()
    elapsed = (state.finished or now) - state.started
    total = state.total
    done = state.done
    fraction = (done / total) if total else 0.0

    # Same trap as the GPU percentage: on the first frame elapsed is tiny, and
    # done/elapsed produces an absurd rate and a nonsense eta.
    rate = done / elapsed if elapsed > 0.001 else 0.0
    if rate > 0 and total > done:
        eta = human_time((total - done) / rate)
    else:
        eta = ""

    head = f"{label:<8} {done:>6,}/{total:<6,}"
    if width >= 60:
        head += f" [{bar(fraction, 20)}] {fraction * 100:3.0f}%"
    if rate > 0:
        head += f"  {rate:,.0f}/s"
    if eta and total:
        head += f"  eta {eta}"

    # Line 2: which files are in flight right now.
    names = state.decoding or state.reading
    detail = "  ".join(short_name(p, root) for p in names[:MAX_NAMES])
    if len(names) > MAX_NAMES:
        detail += f"  (+{len(names) - MAX_NAMES} more)"
    if not detail:
        detail = "idle"

    ring = ""
    if state.ring_cap:
        used, cap = state.ring_used, state.ring_cap
        share = used / cap if cap else 0.0
        ring = f"ring {human_bytes(used)}/{human_bytes(cap)} {share * 100:3.0f}%"

    # GPU time is wall time spent inside forward passes, so it cannot exceed
    # the elapsed window. Clamp anyway: the first frame is drawn microseconds
    # after start(), and dividing by that produced percentages in the millions.
    gpu_pct = 0.0
    if elapsed > 0.001:
        gpu_pct = min(100.0, state.gpu_seconds / elapsed * 100)
    times = f"cpu {human_time(state.cpu_seconds)}   gpu " \
            f"{human_time(state.gpu_seconds)} ({gpu_pct:.0f}%)"
    if state.bytes_read:
        times += f"   read {human_bytes(state.bytes_read)}"
    if state.errors:
        times += f"   errors {state.errors}"

    second = "  ".join(x for x in (detail, ring, times) if x)
    if width and len(head) > width:
        head = head[: width - 1] + "…"
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

    # -- pipeline-facing API (cheap; called from worker threads) ---------------

    def set_total(self, total: int) -> None:
        with self._lock:
            self.state.total = total

    def note_reading(self, path: str) -> None:
        with self._lock:
            self.state.reading.append(path)
            self._trim_locked()

    def note_decoding(self, path: str) -> None:
        with self._lock:
            self.state.decoding.append(path)
            self._trim_locked()

    def note_finished(self, path: str) -> None:
        with self._lock:
            if path in self.state.decoding:
                self.state.decoding.remove(path)
            elif path in self.state.reading:
                self.state.reading.remove(path)

    def note_completed(self) -> None:
        with self._lock:
            self.state.done += 1

    def note_error(self) -> None:
        with self._lock:
            self.state.errors += 1

    def set_ring(self, used: int, cap: int) -> None:
        with self._lock:
            self.state.ring_used = used
            self.state.ring_cap = cap

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
        width = _term_width(self._stream)
        text = render(state, self.root, self.label, width=width)
        # Two lines rewritten in place: carriage return, draw, and clear to
        # end of line so a shrinking line never leaves debris behind.
        pad = "\r" + "\033[K"
        try:
            self._stream.write(pad.join(text.split("\n")) + pad)
            self._stream.flush()
            self._drawn_lines = 2
        except (OSError, ValueError):  # pragma: no cover - closed stdout
            self.enabled = False

    def stop(self) -> None:
        """Finish, leaving the final state on screen with a newline."""
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            self.state.finished = time.perf_counter()
        self.draw()
        try:
            self._stream.write("\n")
            self._stream.flush()
        except (OSError, ValueError):  # pragma: no cover
            pass
        self.enabled = False

    # Context manager so an exception cannot leave the cursor mid-line.
    def __enter__(self) -> "ProgressBar":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


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