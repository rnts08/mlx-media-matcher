"""Score videos against a CLIP query by the single best-matching keyframe.

A video is "about" a query if *any* keyframe matches, so max-pooling over
frames beats averaging the video into one vector (which dilutes a subject that
appears for only part of the clip). The best frame's timestamp comes back too,
so the CLI can point at the moment that matched.
"""

from __future__ import annotations

import numpy as np


def best_moment(
    query: np.ndarray, frames: dict[float, np.ndarray]
) -> tuple[float, float]:
    """Return (max cosine score, timestamp of the best-matching keyframe)."""
    if not frames:
        return 0.0, 0.0
    timestamps = list(frames)
    matrix = np.stack([frames[t] for t in timestamps])
    sims = matrix @ query
    k = int(np.argmax(sims))
    return float(sims[k]), timestamps[k]


def score_video(query: np.ndarray, frames: dict[float, np.ndarray]) -> float:
    return best_moment(query, frames)[0]


def format_timestamp(seconds: float) -> str:
    """0:00 / 1:02:03 style timestamp for the best-matching moment."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
