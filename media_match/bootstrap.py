"""Bootstrap: the fast first pass over a media tree.

Walks the whole tree and records filename, type and size in the index — no
decoding, no hashing, no embeddings. This is what a user runs first to get a
quick inventory and clean up the library before the expensive `scan`.

With flags it can also prune duplicates and non-media files, consolidate
extensions to their canonical form, and fix missing extensions. File
operations are dry-run unless ``--yes`` is given.

Audio is inventoried here (kind ``audio``) so ``--media-only`` never treats
music as junk, but audio is not scanned yet. TODO(audio): add probing,
fingerprinting and interpretation in the scan/analyze stages; see
``docs/FEATURE-audio.md`` and the TODO markers in ``pipeline.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from .config import is_ignored
from .pipeline import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

# ---------------------------------------------------------------------------
# Extension consolidation
# ---------------------------------------------------------------------------

# Any known extension (lower-cased, with dot) -> canonical form. The canonical
# form is the most common lower-case spelling.
CANONICAL_EXTENSIONS: dict[str, str] = {
    # images
    ".jpg": ".jpg", ".jpeg": ".jpg", ".jpe": ".jpg", ".jfif": ".jpg",
    ".png": ".png",
    ".gif": ".gif",
    ".bmp": ".bmp", ".dib": ".bmp",
    ".tif": ".tif", ".tiff": ".tif",
    ".webp": ".webp",
    ".heic": ".heic", ".heif": ".heic",
    # videos
    ".mp4": ".mp4", ".m4v": ".mp4", ".mp4v": ".mp4",
    ".mov": ".mov", ".qt": ".mov",
    ".avi": ".avi",
    ".mkv": ".mkv",
    ".webm": ".webm",
    ".mpg": ".mpg", ".mpeg": ".mpg", ".mp2": ".mpg",
    ".mpe": ".mpg", ".mpv": ".mpg",
    ".wmv": ".wmv",
    # audio: canonical spelling is the extension itself (TODO(audio): move all
    # three kind sets into one media registry; see docs/FEATURE-audio.md)
    **{ext: (".ogg" if ext == ".oga" else ext) for ext in AUDIO_EXTENSIONS},
}

# All extensions we consider "media" (image, video, audio).
MEDIA_EXTENSIONS = frozenset(CANONICAL_EXTENSIONS)


def canonical_extension(ext: str) -> str | None:
    """Canonical form of an extension, or None if it is not a known media ext."""
    return CANONICAL_EXTENSIONS.get(ext.lower())


def classify_extension(ext: str) -> str:
    """'image', 'video', 'audio' or 'other' for an extension (with or without dot)."""
    ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "other"


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------


def walk_tree(root: str, *, include_video: bool = True,
              ignore: list[str] | None = None) -> list[tuple[str, int, float, str, str]]:
    """Every file under root as (path, size, mtime, ext, kind).

    Unlike iter_media this does not filter to media — bootstrap inventories
    the whole tree so --media-only can prune the non-media files.
    """
    ignore = ignore or []
    rows: list[tuple[str, int, float, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            if ignore and is_ignored(os.path.relpath(path, root), ignore):
                continue
            ext = os.path.splitext(name)[1].lower()
            kind = classify_extension(ext)
            if kind == "other" and not include_video:
                # still record it; --media-only decides what to prune
                pass
            try:
                st = os.stat(path)
            except OSError:
                continue
            rows.append((path, st.st_size, st.st_mtime, ext, kind))
    return rows


# ---------------------------------------------------------------------------
# Extension consolidation / fixing
# ---------------------------------------------------------------------------


def consolidated_path(path: str) -> str | None:
    """New path with the extension in canonical form, or None if unchanged."""
    ext = os.path.splitext(path)[1]
    canonical = canonical_extension(ext)
    if canonical is None or canonical == ext:
        return None
    return path[: -len(ext)] + canonical


def _unique_path(path: str) -> str:
    """path, or path with a numeric suffix if it already exists."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}.{n}{ext}"):
        n += 1
    return f"{stem}.{n}{ext}"


def detect_media_type(path: str) -> str | None:
    """Actual media type of a file from its content: '.jpg', '.mp4', etc.

    Returns None when the file cannot be identified as a known media type.
    """
    # images via PIL
    try:
        from PIL import Image

        with Image.open(path) as img:
            fmt = (img.format or "").lower()
        mapping = {
            "jpeg": ".jpg", "png": ".png", "gif": ".gif", "bmp": ".bmp",
            "tiff": ".tif", "webp": ".webp", "heic": ".heic", "heif": ".heic",
        }
        if fmt in mapping:
            return mapping[fmt]
    except Exception:  # noqa: BLE001
        pass
    # videos/audio via ffprobe. TODO(audio): return the actual audio container
    # where it is known (.m4a/.flac/.wav) instead of always .mp3, and read
    # music tags here only if it stays cheap; the full tag/fingerprint pass
    # belongs in scan --audio (docs/FEATURE-audio.md).
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name", "-of", "json", path],
            capture_output=True, text=True, timeout=10,
        ).stdout
        import json

        data = json.loads(out)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                return ".mp4"  # canonical video container
            if stream.get("codec_type") == "audio":
                return ".mp3"  # canonical audio
    except Exception:  # noqa: BLE001
        pass
    return None


def fixed_extension_path(path: str) -> str | None:
    """New path with a missing/unknown extension fixed, or None if unchanged."""
    ext = os.path.splitext(path)[1]
    if ext and canonical_extension(ext) is not None:
        return None  # already has a known media extension
    detected = detect_media_type(path)
    if detected is None:
        return None
    return path + detected


# ---------------------------------------------------------------------------
# Duplicate finding
# ---------------------------------------------------------------------------


def find_exact_duplicates(rows: list[tuple[str, int, float, str, str]]) -> list[list[int]]:
    """Group indices of rows that are exact duplicates (same sha1).

    Only same-size candidates are hashed, so this stays fast on large trees.
    Returns a list of groups (each a list of row indices), sorted newest-first
    within each group.
    """
    import hashlib

    by_size: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_size.setdefault(row[1], []).append(i)

    groups: list[list[int]] = []
    for indices in by_size.values():
        if len(indices) < 2:
            continue
        by_sha1: dict[str, list[int]] = {}
        for i in indices:
            path = rows[i][0]
            try:
                h = hashlib.sha1()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                by_sha1.setdefault(h.hexdigest(), []).append(i)
            except OSError:
                continue
        for sha1_indices in by_sha1.values():
            if len(sha1_indices) >= 2:
                # newest first (mtime desc)
                sha1_indices.sort(key=lambda i: -rows[i][2])
                groups.append(sha1_indices)
    return groups


def find_phash_duplicates(rows: list[tuple[str, int, float, str, str]]) -> list[list[int]]:
    """Group indices of image rows that are perceptual duplicates (pHash ==).

    More expensive than find_exact_duplicates (decodes each image) but finds
    re-encodes and rescales that sha1 misses. Only images are considered.
    """
    from .hashing import hash_to_hex, phash
    from .pipeline import _open_image_tolerant

    by_size: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        if row[4] != "image":
            continue
        by_size.setdefault(row[1], []).append(i)

    groups: list[list[int]] = []
    for indices in by_size.values():
        if len(indices) < 2:
            continue
        by_phash: dict[str, list[int]] = {}
        for i in indices:
            path = rows[i][0]
            try:
                img = _open_image_tolerant(path)
                try:
                    h = hash_to_hex(phash(img))
                finally:
                    img.close()
                by_phash.setdefault(h, []).append(i)
            except Exception:  # noqa: BLE001
                continue
        for phash_indices in by_phash.values():
            if len(phash_indices) >= 2:
                phash_indices.sort(key=lambda i: -rows[i][2])
                groups.append(phash_indices)
    return groups
