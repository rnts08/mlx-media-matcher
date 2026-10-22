"""Perceptual hashing (pHash) for resolution-independent duplicate detection.

pHash is computed from the DCT of a canonical 32x32 grayscale version of the
image, so a 4000px photo and a 150px thumbnail of the same picture produce
nearly identical 64-bit hashes.

TODO(audio): audio fingerprints are a different primitive. Chromaprint returns
a variable-length sequence over time, matched by alignment/bit-error rate, not
a 64-bit hamming comparison. Add an audio fingerprint module (or adapter around
`fpcalc -raw`) rather than reusing ``phash``/``hamming``. See
``docs/FEATURE-audio.md``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_DCT_SIZE = 32
_HASH_SIZE = 8


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis matrix."""
    k = np.arange(n)
    c = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    c[0] *= np.sqrt(1 / n)
    c[1:] *= np.sqrt(2 / n)
    return c


_DCT = _dct_matrix(_DCT_SIZE)


def phash(img: Image.Image) -> int:
    """64-bit perceptual hash of a PIL image."""
    gray = img.convert("L").resize((_DCT_SIZE, _DCT_SIZE), Image.Resampling.LANCZOS)
    x = np.asarray(gray, dtype=np.float64)
    dct = _DCT @ x @ _DCT.T
    block = dct[:_HASH_SIZE, :_HASH_SIZE].flatten()
    median = np.median(block[1:])  # skip DC component
    bits = block > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hash_to_hex(value: int) -> str:
    return f"{value:016x}"


def hex_to_hash(text: str) -> int:
    return int(text, 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hamming_pairs(hashes_hex: list[str], threshold: int) -> list[tuple[int, int, int]]:
    """All (i, j, distance) pairs with i < j and distance <= threshold.

    Vectorized with numpy popcount, processed in row chunks so memory stays
    bounded for large collections.
    """
    if len(hashes_hex) < 2:
        return []
    arr = np.array([hex_to_hash(h) for h in hashes_hex], dtype=np.uint64)
    n = len(arr)
    chunk = max(1, min(n, 4096 * 1024 // max(n, 1) or 1))
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        xor = arr[start:stop, None] ^ arr[None, :]
        dist = np.bitwise_count(xor)
        # upper triangle only, within this row block
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                pairs.append((i, j, int(dist[r, c])))
    return pairs
