"""Per-file image diagnostics.

Pillow reports damage through ``warnings.warn``: a truncated read, a malformed
MPO container, an image past its decompression-bomb guard. Those land on stderr
with a ``site-packages/PIL/...`` path, once per file, which is noise when you
are scanning twenty thousand photos — and useless, because they never say
*which* file was wrong.

This module captures the warnings instead of leaking them, works out what each
one actually means for the file, and hands the caller a sentence it can print
next to a filename:

    scan:  Photos/Sunset.mpo  malformed MPO container, read as plain JPEG
    check: ~/Pictures/IMG_0042.jpg  truncated file, 640x480 header intact

Only ``truncated`` is worth *acting* on, and ``media-match check`` repairs it.
The others are noted and left alone: a malformed MPO still decodes as a JPEG,
and a very large image is not broken.

The status constants live here rather than in ``check`` so that both can use
them without importing each other.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

# What diagnose_image returns. These are the same strings `check` has always
# returned, so the database and the CLI keep working unchanged.
OK = "ok"
TRUNCATED = "truncated"
CORRUPT = "corrupt"

# Readable, but worth a mention. Never acted on destructively.
NOTABLE = "notable"


@dataclass(frozen=True)
class Diagnosis:
    """What is wrong with one file, if anything."""

    status: str
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def actionable(self) -> bool:
        """True when 'media-match check' can do something useful about it."""
        return self.status == TRUNCATED


def _truncated_text(msg: str) -> bool:
    return "truncated" in msg.lower()


def _mpo_text(msg: str) -> bool:
    return "malformed mpo" in msg.lower()


def _bomb_text(msg: str) -> bool:
    return "decompression bomb" in msg.lower()


def _bomb_message(msg: str) -> str:
    """'Image size (108000000 pixels) exceeds limit of 89478485 pixels' -> '108 MP'.

    The image size comes first and the limit second, so take the first long
    number and stop. Strip the brackets first: PIL writes it as '(108000000',
    which is not a bare digit and would otherwise be skipped, leaving the
    limit to be reported as the size.
    """
    numbers = re.findall(r"\d{4,}", msg)
    size = numbers[0] if numbers else ""
    if not size:
        return "image is larger than Pillow's decompression-bomb limit"
    mpx = int(size) / 1_000_000
    return (f"image is {mpx:.0f} MP, past Pillow's decompression-bomb limit — "
            f"decoded anyway, nothing is damaged")


def classify(messages: list[str], *, width: int = 0, height: int = 0) -> Diagnosis:
    """Turn Pillow's warning texts into one status and one sentence.

    Bomb and MPO warnings come first: a truncated file can also trip them, and
    the more specific problem is the more useful thing to say.
    """
    for msg in messages:
        if _bomb_text(msg):
            return Diagnosis(NOTABLE, _bomb_message(msg))
    for msg in messages:
        if _mpo_text(msg):
            return Diagnosis(NOTABLE,
                             "malformed MPO container, read as a plain JPEG — "
                             "the image itself is fine")
    for msg in messages:
        if _truncated_text(msg):
            where = f"{width}x{height} header intact but the pixel data ends early" \
                if width and height else "the pixel data ends early"
            return Diagnosis(
                TRUNCATED,
                f"truncated file — {where}; 'media-match check' can re-encode it",
            )
    return Diagnosis(OK)


def _decode(path: str, *, tolerant: bool) -> tuple[bool, tuple[int, int], list[str]]:
    """Decode once; return (ok, size, warning texts). Never raises."""
    from PIL import Image, ImageFile

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # LOAD_TRUNCATED_IMAGES is a global, but the race with the scan's own
        # worker threads is benign: it only affects files that are truncated.
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = tolerant
        try:
            with Image.open(path) as img:
                size = img.size
                img.load()
            return True, size, [str(w.message) for w in caught]
        except (OSError, IOError, ValueError):
            return False, (0, 0), [str(w.message) for w in caught]
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def diagnose_image(path: str) -> Diagnosis:
    """Open and decode ``path``, reporting what Pillow complained about.

    Tolerates truncation the same way the scan does, so a file the scan can
    index is never called corrupt here.
    """
    ok, size, messages = _decode(path, tolerant=False)
    if ok:
        return classify(messages, width=size[0], height=size[1])

    # It needed tolerance: the file is damaged, but the readable part works.
    ok, size, tolerant_messages = _decode(path, tolerant=True)
    if ok:
        return classify([*tolerant_messages, "Truncated File Read"],
                        width=size[0], height=size[1])
    return Diagnosis(CORRUPT, "Pillow cannot identify or decode this file")