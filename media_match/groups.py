"""Duplicate grouping: cluster files whose content or perceptual hash matches.

Two images are linked when they are byte-identical (same sha1) or visually
near-identical (pHash hamming distance <= threshold). pHash is resolution
independent, so thumbnails group with their full-resolution originals. Within
a group, the largest image (by pixel count) is the "best version" (primary);
everything else is a duplicate/variant.

`find_similarity_groups` groups more loosely via CLIP embedding cosine
similarity: it also links different photos of the same subject, not only
re-encodes of the same photo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .db import FileRecord
from .hashing import hamming_pairs, hex_to_hash


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class DuplicateGroup:
    best: FileRecord
    others: list[tuple[FileRecord, int]] = field(default_factory=list)  # (record, dist to best)

    @property
    def size(self) -> int:
        return len(self.others) + 1


def _megapixels(r: FileRecord) -> int:
    return r.width * r.height


def _taken_key(r: FileRecord) -> tuple[str, float]:
    """Sort key for ``prefer="date"``: newest capture first (mtime tiebreak)."""
    return (getattr(r, "captured_at", "") or "", getattr(r, "mtime", 0.0))


def find_duplicate_groups(
    records: list[FileRecord], threshold: int = 8, prefer: str = "resolution",
    exact_phash: bool = False
) -> list[DuplicateGroup]:
    n = len(records)
    if n == 0:
        return []
    uf = _UnionFind(n)

    # byte-identical files
    by_sha1: dict[str, int] = {}
    for i, r in enumerate(records):
        if r.sha1 in by_sha1:
            uf.union(by_sha1[r.sha1], i)
        else:
            by_sha1[r.sha1] = i

    if exact_phash:
        # fast path: group files with identical perceptual hash (Hamming distance = 0)
        phash_groups: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            ph_key = r.phash  # already hex string from hashing
            phash_groups.setdefault(ph_key, []).append(i)
        # Only keep groups with 2+ files that share the exact same phash
        for ph_key, members in phash_groups.items():
            if len(members) >= 2:
                best_idx = members[0]
                others = [
                    (records[i], 0)  # Hamming distance 0 since phash is identical
                    for i in members
                    if i != best_idx
                ]
                others.sort(key=lambda t: -_megapixels(records[i]))
                groups.append(DuplicateGroup(best=records[best_idx], others=others))
        # Fall through to the normal Hamming-distance-based grouping for all other files
        hashes = [r.phash for r in records]
        for i, j, _dist in hamming_pairs(hashes, threshold):
            uf.union(i, j)
    else:
        # visually near-identical files (resolution independent)
        hashes = [r.phash for r in records]
        for i, j, _dist in hamming_pairs(hashes, threshold):
            uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    groups: list[DuplicateGroup] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        if prefer == "date":
            best_idx = max(members, key=lambda i: (
                _taken_key(records[i]), _megapixels(records[i]), records[i].size
            ))
        else:
            best_idx = max(members, key=lambda i: (
                _megapixels(records[i]), records[i].size
            ))
        best_hash = hex_to_hash(records[best_idx].phash)
        others = [
            (records[i], (hex_to_hash(records[i].phash) ^ best_hash).bit_count())
            for i in members
            if i != best_idx
        ]
        others.sort(key=lambda t: -_megapixels(t[0]))
        groups.append(DuplicateGroup(best=records[best_idx], others=others))

    groups.sort(key=lambda g: -g.size)
    return groups


@dataclass
class SimilarGroup:
    best: FileRecord
    others: list[tuple[FileRecord, float]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.others) + 1


def cluster_by_similarity(embeddings: np.ndarray, threshold: float = 0.85) -> list[int]:
    """Connected components of the cosine-similarity graph.

    Returns a cluster label per row in ``embeddings`` (0-based, ordered by
    first appearance). Unlike ``find_similarity_groups`` this keeps singletons,
    which is what people/photo clustering wants: every face crop belongs to
    exactly one cluster even if it resembles nobody else.
    """
    n = len(embeddings)
    if n == 0:
        return []
    uf = _UnionFind(n)
    chunk = max(1, min(n, 4096 * 1024 // max(n, 1) or 1))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = embeddings[start:stop] @ embeddings.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                uf.union(i, j)
    labels: list[int] = []
    roots: dict[int, int] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels


def find_similarity_groups(
    records: list[FileRecord], embeddings: np.ndarray, threshold: float = 0.85
) -> list[SimilarGroup]:
    """Group files whose CLIP embeddings are cosine-similar >= threshold."""
    n = len(records)
    if n < 2:
        return []
    uf = _UnionFind(n)
    chunk = max(1, min(n, 4096 * 1024 // max(n, 1) or 1))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = embeddings[start:stop] @ embeddings.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    groups: list[SimilarGroup] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        best_idx = max(members, key=lambda i: (_megapixels(records[i]), records[i].size))
        best_vec = embeddings[best_idx]
        others = [
            (records[i], float(embeddings[i] @ best_vec))
            for i in members
            if i != best_idx
        ]
        others.sort(key=lambda t: -_megapixels(t[0]))
        groups.append(SimilarGroup(best=records[best_idx], others=others))

    groups.sort(key=lambda g: -g.size)
    return groups
