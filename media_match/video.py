"""Video probing and keyframe sampling via the system ffmpeg/ffprobe.

A video is reduced to a sparse, *time-uniform* set of keyframes — about one per
second, capped for long clips. Those frames then go through the exact Phase-1
image pipeline (pHash + CLIP + optional VLM).

Sampling on a fixed time grid is what makes duplicate comparison cheap: two
copies of the same clip (a re-encode, a lower resolution) produce frames at the
same timestamps, so matching is index-aligned instead of all-pairs. The grid is
derived only from the duration and the frame count, so it is stable across
resolutions and encoders.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

FRAME_SIZE = 224          # square; pHash crops to 32px, CLIP takes 224
DEFAULT_INTERVAL = 1.0    # target seconds between keyframes (~1 fps)
MIN_FRAMES = 3            # always take at least this many from a real clip
MAX_FRAMES = 120          # long clips sample sparser than 1 fps


@dataclass
class VideoMeta:
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int


def _fraction(text: str | None) -> float:
    if not text:
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(text)


def probe(path: str) -> VideoMeta:
    """Read container/stream metadata with ffprobe."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError(f"no video stream in {path}")
    v = streams[0]
    fmt = data.get("format", {})
    return VideoMeta(
        duration=float(v.get("duration") or fmt.get("duration") or 0.0),
        width=int(v.get("width") or 0),
        height=int(v.get("height") or 0),
        fps=_fraction(v.get("avg_frame_rate")),
        codec=v.get("codec_name") or "",
        bitrate=int(v.get("bit_rate") or fmt.get("bit_rate") or 0),
    )


def sample_count(
    duration: float,
    interval: float = DEFAULT_INTERVAL,
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_FRAMES,
) -> int:
    """How many keyframes to take from a clip of this many seconds.

    ~1 frame per ``interval`` seconds, floored at ``min_frames`` (so very short
    clips still get a few samples) and capped at ``max_frames`` (so a long clip
    samples sparser than 1 fps rather than exploding the frame count).
    """
    if duration <= 0:
        return 0
    n = math.ceil(duration / interval)
    return max(min_frames, min(max_frames, n))


def _run(cmd: list[str], what: str) -> bytes:
    """Run ffmpeg and return stdout, or raise a short, readable error.

    ``check=True`` would surface the whole command line; the last stderr line
    ("No such file or directory", "Invalid data found", ...) is far more useful
    in a scan log.
    """
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = err[-1] if err else f"exit status {proc.returncode}"
        raise RuntimeError(f"{what} failed: {detail}")
    return proc.stdout


def extract_frames(
    path: str,
    meta: VideoMeta,
    count: int | None = None,
    frame_size: int = FRAME_SIZE,
) -> list[tuple[float, np.ndarray]]:
    """Sample ``count`` keyframes as ``[(timestamp, HxWx3 uint8), ...]``.

    A single ffmpeg pass scales every frame to ``frame_size`` x ``frame_size``
    and pipes raw RGB; the fixed size means the stream can be split by simple
    arithmetic (no per-frame container/PNG parsing). Timestamps are the grid
    ``k * duration / n``, which is what lets two copies align frame-for-frame.
    """
    if count is None:
        count = sample_count(meta.duration)
    if count <= 0 or meta.duration <= 0:
        return []

    fps = count / meta.duration
    raw = _run(
        [FFMPEG, "-v", "error", "-nostdin", "-i", path,
         "-vf", f"fps={fps:.6f},scale={frame_size}:{frame_size}",
         "-frames:v", str(count), "-an",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        what=f"decoding {os.path.basename(path)}",
    )

    frame_bytes = frame_size * frame_size * 3
    n = len(raw) // frame_bytes
    frames: list[tuple[float, np.ndarray]] = []
    for k in range(n):
        arr = np.frombuffer(raw, np.uint8, frame_bytes, k * frame_bytes)
        frames.append((k * meta.duration / n, arr.reshape(frame_size, frame_size, 3)))
    return frames


def extract_still_png(path: str, at: float, size: int = 448) -> bytes:
    """A single PNG frame at ``at`` seconds — used to caption a video."""
    return _run(
        [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{max(at, 0):.3f}", "-i", path,
         "-frames:v", "1", "-an", "-vf", f"scale={size}:{size}",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        what=f"frame extraction from {os.path.basename(path)}",
    )

