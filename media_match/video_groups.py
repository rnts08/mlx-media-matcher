"""Group videos by keyframe content — cheaply.

Because every video is sampled on a uniform time grid, two copies of the same
clip line up by *relative position*: frame *i* of one corresponds to roughly
frame *i* of the other. So duplicates are found by comparing a frame with its
counterpart (plus a tiny +/-band for timing jitter), never the full O(n^2)
grid. A trimmed clip lines up by proportional position and still matches its
parent.

`find_video_duplicate_groups` uses frame pHashes (resolution/encoder
independent, like the image path). `find_video_similarity_groups` uses one
mean-pooled CLIP vector per video, so semantic grouping stays O(videos^2) even
when each video has a hundred keyframes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .db import VideoRecord
from .groups import DuplicateGroup, SimilarGroup, _UnionFind

MIN_MATCHES = 2  # a couple of matching frames is coincidence; more is a copy


def pair_ratio(
    a_hex: list[str],
    b_hex: list[str],
    threshold: int = 12,
    band: int = 2,
) -> tuple[float, int]:
    """Fraction of aligned keyframes that match, and the best hamming distance.

    Frames are compared by relative position (``linspace`` over the longer
    sequence) with a +/-``band`` window, so this is O(n) rather than O(n^2).
    """
    if not a_hex or not b_hex:
        return 0.0, 64
    if len(a_hex) > len(b_hex):
        a_hex, b_hex = b_hex, a_hex
    na, nb = len(a_hex), len(b_hex)

    a = np.array([int(h, 16) for h in a_hex], dtype=np.uint64)
    b = np.array([int(h, 16) for h in b_hex], dtype=np.uint64)
    pos = np.round(np.linspace(0, nb - 1, na)).astype(int)

    best = np.full(na, 64, dtype=np.int64)
    for off in range(-band, band + 1):
        idx = np.clip(pos + off, 0, nb - 1)
        dist = np.bitwise_count(a ^ b[idx])
        best = np.minimum(best, dist)

    matches = int((best <= threshold).sum())
    return matches / na, int(best.min())


def _best_key(record: VideoRecord) -> tuple[int, int, int]:
    return (record.width * record.height, record.bitrate, record.size)


def find_video_duplicate_groups(
    records: list[VideoRecord],
    frame_threshold: int = 12,
    merge_ratio: float = 0.6,
    band: int = 2,
) -> list[DuplicateGroup]:
    """Cluster duplicate/re-encode videos by aligned keyframe pHash matching."""
    n = len(records)
    if n == 0:
        return []
    uf = _UnionFind(n)

    by_sha1: dict[str, int] = {}
    for i, r in enumerate(records):
        if r.sha1 in by_sha1:
            uf.union(by_sha1[r.sha1], i)
        else:
            by_sha1[r.sha1] = i

    for i in range(n):
        for j in range(i + 1, n):
            ratio, _ = pair_ratio(
                records[i].phashes, records[j].phashes,
                threshold=frame_threshold, band=band,
            )
            if ratio >= merge_ratio:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    groups: list[DuplicateGroup] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        best_idx = max(members, key=lambda i: _best_key(records[i]))
        others = []
        for i in members:
            if i == best_idx:
                continue
            _ratio, dist = pair_ratio(
                records[best_idx].phashes, records[i].phashes,
                threshold=frame_threshold, band=band,
            )
            others.append((records[i], dist))
        others.sort(key=lambda t: -_best_key(t[0])[0])
        groups.append(DuplicateGroup(best=records[best_idx], others=others))

    groups.sort(key=lambda g: -g.size)
    return groups


def _mean_vector(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.zeros((0,), dtype=np.float32)
    mean = frames.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else mean


def find_video_similarity_groups(
    records: list[VideoRecord],
    embeddings: dict[int, np.ndarray],
    threshold: float = 0.85,
) -> list[SimilarGroup]:
    """Group videos whose mean-pooled CLIP keyframe vectors are similar."""
    indexed = [r for r in records if r.id in embeddings and embeddings[r.id].size]
    n = len(indexed)
    if n < 2:
        return []
    means = np.stack([_mean_vector(embeddings[r.id]) for r in indexed])
    uf = _UnionFind(n)
    sims = means @ means.T
    rows, cols = np.nonzero(sims >= threshold)
    for r, c in zip(rows, cols):
        if r < c:
            uf.union(int(r), int(c))

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    groups: list[SimilarGroup] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        best_idx = max(members, key=lambda i: _best_key(indexed[i]))
        best_vec = means[best_idx]
        others = [
            (indexed[i], float(means[i] @ best_vec))
            for i in members
            if i != best_idx
        ]
        others.sort(key=lambda t: -_best_key(t[0])[0])
        groups.append(SimilarGroup(best=indexed[best_idx], others=others))

    groups.sort(key=lambda g: -g.size)
    return groups
