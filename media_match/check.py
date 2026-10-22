"""Image loadability check and interactive repair.

Some files in a real photo library are not what they seem: a JPEG whose last
bytes were lost in a copy (truncated — PIL can still decode the valid part
with ``LOAD_TRUNCATED_IMAGES``), or a file that is not an image at all
(corrupt — PIL cannot even identify the format).

The scan pipeline tolerates truncation (see ``pipeline._open_image_tolerant``)
so truncated photos are still indexed. This module answers the follow-up
question — *which* files are affected, and shall we fix them? — and offers
interactive repair:

  * truncated  -> re-encode the decodable part (the original is kept as
                  ``<name>.truncated``), so the file loads cleanly everywhere
  * corrupt    -> move to a quarantine folder (never deleted outright)
"""

from __future__ import annotations

import os
import shutil

from .diagnose import CORRUPT, NOTABLE, OK, TRUNCATED, diagnose_image
from .pipeline import IMAGE_EXTENSIONS, iter_media

# TODO(audio): diagnose_tree() should also probe audio (ffprobe metadata plus a
# short decode) and report it with the same ok/truncated/corrupt/notable shape.
# See docs/FEATURE-audio.md and `check --audio`.


def check_image(path: str) -> str:
    """Classify one image file: 'ok', 'truncated', 'corrupt' or 'notable'.

    'truncated' means PIL needs ``LOAD_TRUNCATED_IMAGES`` to decode it — the
    file is damaged but the recoverable part is usable. 'corrupt' means PIL
    cannot identify or decode the file at all. 'notable' means the file loads
    fine but Pillow flagged it (a malformed MPO container, an image past its
    decompression-bomb limit); it is reported, never repaired.
    """
    return diagnose_image(path).status


def diagnose_tree(root: str, *, include_video: bool = False,
                  ignore: list[str] | None = None,
                  progress=None) -> dict[str, list[tuple[str, str]]]:
    """Classify every image (and, optionally, video) under ``root``.

    Returns ``{"ok": [...], "truncated": [...], ...}`` as
    ``(absolute_path, explanation)`` pairs, in tree order. ``progress(done,
    path)`` is called after each file, so a long check can show it is alive.
    """
    from .config import is_ignored

    ignore = ignore or []
    out: dict[str, list[tuple[str, str]]] = {
        OK: [], TRUNCATED: [], CORRUPT: [], NOTABLE: [],
    }
    for done, (kind, path) in enumerate(iter_media(root, include_video,
                                                   ignore=ignore), 1):
        if kind == "video":
            # videos are probed with ffmpeg during the scan; a file that
            # ffprobe cannot read is corrupt
            from .video import probe

            try:
                probe(path)
                out[OK].append((path, ""))
            except Exception as exc:  # noqa: BLE001 - any probe failure is corrupt
                out[CORRUPT].append((path, f"ffprobe cannot read it: {exc}"))
        else:
            found = diagnose_image(path)
            out[found.status].append((path, found.message))
        if progress is not None:
            progress(done, path)
    return out


def repair_truncated(path: str, *, backup_suffix: str = ".truncated") -> str:
    """Re-encode a truncated image so it loads cleanly everywhere.

    The original bytes are kept as ``path + backup_suffix`` (an existing
    backup is never overwritten). Returns a short human-readable result.
    """
    import tempfile

    import warnings

    from PIL import Image, ImageFile

    backup = path + backup_suffix
    if os.path.exists(backup):
        return f"backup exists, skipped: {backup}"

    # Re-encoding is *supposed* to read a truncated file, so the warning it
    # provokes is expected and would only confuse the user.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as img:
                img.load()
                # re-encode from the decoded pixels; the truncated tail is lost
                # but the file becomes complete and loadable. Write to a temp
                # file first so the original is never at risk.
                fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(path)[1],
                                           dir=os.path.dirname(path))
                os.close(fd)
                try:
                    img.save(tmp)
                except Exception:
                    os.unlink(tmp)
                    raise
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
    os.replace(path, backup)
    os.replace(tmp, path)
    return f"repaired (backup: {backup})"


def quarantine(path: str, quarantine_dir: str) -> str:
    """Move a corrupt file into ``quarantine_dir``, preserving its name."""
    os.makedirs(quarantine_dir, exist_ok=True)
    dest = os.path.join(quarantine_dir, os.path.basename(path))
    n = 1
    while os.path.exists(dest):
        stem, ext = os.path.splitext(os.path.basename(path))
        dest = os.path.join(quarantine_dir, f"{stem}.{n}{ext}")
        n += 1
    shutil.move(path, dest)
    return dest
