"""Terminal output shared by commands, model loading and progress displays.

Everything here is deliberately small and dependency-free.  The rules are:

* stdout carries command output; status and progress go to stderr.
* a non-UTF-8 or non-terminal stderr still gets usable one-line status, never
  a traceback from a block character.
* model weights are announced *before* a progress bar owns the terminal, so a
  first-run download cannot look like a hung stage.
"""

from __future__ import annotations

import os
import sys
import time

from .progress import Spinner


def status(line: str) -> None:
    """Write one status line, even to a stream that cannot encode the bar.

    A non-UTF-8 stderr gets an ASCII approximation instead of raising inside a
    worker thread and silencing every later message.
    """
    try:
        print(line, file=sys.stderr)
        sys.stderr.flush()
    except UnicodeEncodeError:
        try:
            sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
            sys.stderr.flush()
        except (OSError, ValueError):  # pragma: no cover - closed stderr
            pass
    except (OSError, ValueError):  # pragma: no cover - closed stderr
        pass


def fmt_bytes(n: float) -> str:
    """Human-readable byte size: 512.0B / 3.4MB / 1.2GB."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def fmt_seconds(s: float) -> str:
    """Human duration: 0s / 3.2s / 4m12s / 2h05m."""
    if s <= 0:
        return "0s"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def progress_enabled(args) -> bool:
    """Whether this run may draw a live display on stderr.

    ``args`` may be None for call sites (model loading) that do not have the
    parsed namespace.  An explicit ``--no-progress`` or the environment always
    wins; otherwise stderr must be a terminal.
    """
    if args is not None and getattr(args, "no_progress", False):
        return False
    if os.environ.get("MEDIA_MATCH_NO_PROGRESS"):
        return False
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - detached stderr
        return False


def add_progress_flag(parser) -> None:
    parser.add_argument(
        "--no-progress", action="store_true",
        help="no live progress display (implied when stderr is not a terminal)",
    )


def load_model(model_holder, what: str = "CLIP model", *,
               enabled: bool | None = None) -> None:
    """Load model weights before a progress bar starts, and say so.

    Loading is lazy in the model classes.  Calling this first turns a first-run
    download or kernel compile from an unexplained pause into a named step.
    On a terminal the wait is animated; piped logs get the same two plain
    lines they have always had.
    """
    ensure = getattr(model_holder, "ensure_loaded", None)
    if not callable(ensure):
        return  # a stand-in in tests has nothing to load
    if getattr(model_holder, "_media_match_model_ready", False):
        return
    model = getattr(model_holder, "model_id", what)
    message = f"loading {model} (first run downloads it) ..."
    if enabled is None:
        enabled = progress_enabled(None)

    started = time.perf_counter()
    if enabled:
        spinner = Spinner(message).start()
        try:
            ensure()
        except BaseException:
            spinner.stop(f"! {model} failed to load")
            raise
        _mark_ready(model_holder)
        spinner.stop(f"✓ model ready in {time.perf_counter() - started:.1f}s")
        return

    status(message)
    ensure()
    _mark_ready(model_holder)
    status(f"  model ready in {time.perf_counter() - started:.1f}s")


def _mark_ready(model_holder) -> None:
    try:
        setattr(model_holder, "_media_match_model_ready", True)
    except (AttributeError, TypeError):  # pragma: no cover - exotic stand-in
        pass
