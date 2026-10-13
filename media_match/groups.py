"""Duplicate grouping: cluster files whose content or perceptual hash matches.

Two images are linked when they are byte-identical (same sha1) or visually
near-identical (pHash hamming distance <= threshold). pHash is resolution
independent, so thumbnails group with their full-resolution originals. Within
a group, the largest image (by pixel count) is the "best version" (primary);
everything else is a duplicate/variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def find_duplicate_groups(
    records: list[FileRecord], threshold: int = 8
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
        best_idx = max(members, key=lambda i: (_megapixels(records[i]), records[i].size))
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
