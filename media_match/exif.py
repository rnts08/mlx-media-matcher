"""Minimal EXIF reader for the index: capture time, camera and GPS.

Deliberately dependency-free: it uses Pillow's ``Image.getexif()`` and the
standard EXIF tag numbers, so it works for JPEG/TIFF/HEIC (via pillow-heif)
without pulling in ``exifread``. Missing or malformed tags never raise — the
caller just gets blanks and falls back to the file's mtime for ordering.
"""

from __future__ import annotations

from datetime import datetime

_EXIF_IFD = 0x8769  # tag -> Exif sub-IFD
_GPS_IFD = 0x8825   # tag -> GPS sub-IFD

# Standard EXIF tag ids.
_DATETIME_ORIGINAL = 36867
_DATETIME_DIGITIZED = 36868
_DATETIME = 306
_MAKE = 271
_MODEL = 272


def _to_degrees(dms, ref) -> float | None:
    try:
        deg, minutes, seconds = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    value = deg + minutes / 60.0 + seconds / 3600.0
    if str(ref).upper() in ("S", "W"):
        value = -value
    return value


def read_exif(img) -> dict:
    """Return ``{captured_at, camera, gps_lat, gps_lon}`` from a PIL image.

    ``captured_at`` is local ``YYYY-MM-DDTHH:MM:SS`` (matching the index's sort
    order) or ``""`` when the file carries no capture time.
    """
    out: dict = {"captured_at": "", "camera": "", "gps_lat": None, "gps_lon": None}
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001 - any decode problem is just "no metadata"
        return out
    if not exif:
        return out

    try:
        sub = exif.get_ifd(_EXIF_IFD)
    except Exception:  # noqa: BLE001
        sub = {}
    raw = sub.get(_DATETIME_ORIGINAL) or sub.get(_DATETIME_DIGITIZED) \
        or exif.get(_DATETIME_ORIGINAL) or exif.get(_DATETIME_DIGITIZED) \
        or exif.get(_DATETIME)
    if raw:
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                out["captured_at"] = datetime.strptime(str(raw).strip(), fmt).strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
                break
            except ValueError:
                continue

    make = str(exif.get(_MAKE) or "").strip()
    model = str(exif.get(_MODEL) or "").strip()
    if make and model.lower().startswith(make.lower()):
        out["camera"] = model
    else:
        out["camera"] = " ".join(part for part in (make, model) if part)

    try:
        gps = exif.get_ifd(_GPS_IFD)
    except Exception:  # noqa: BLE001
        gps = {}
    if gps:
        out["gps_lat"] = _to_degrees(gps.get(2), gps.get(1))
        out["gps_lon"] = _to_degrees(gps.get(4), gps.get(3))
    return out
