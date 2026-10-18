"""The loadability check: classify ok/truncated/corrupt, repair, quarantine."""

import os

import pytest
from PIL import Image

from media_match.check import (
    CORRUPT,
    NOTABLE,
    OK,
    TRUNCATED,
    check_image,
    diagnose_tree,
    quarantine,
    repair_truncated,
)


def _write_image(path, size=(64, 48), color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _truncate(path, drop=225):
    """Cut bytes off the end of a valid JPEG to make it truncated."""
    data = path.read_bytes()
    path.write_bytes(data[:-drop])


def test_check_image_ok(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p)
    assert check_image(str(p)) == OK


def test_check_image_truncated(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p, size=(400, 300))
    _truncate(p, drop=225)
    assert check_image(str(p)) == TRUNCATED


def test_check_image_corrupt(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"this is not an image at all")
    assert check_image(str(p)) == CORRUPT


def test_diagnose_tree_categorizes(tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "ok.png")
    truncated = root / "trunc.jpg"
    _write_image(truncated, size=(400, 300))
    _truncate(truncated, drop=225)
    corrupt = root / "bad.png"
    corrupt.write_bytes(b"not an image")

    result = diagnose_tree(str(root))
    assert [p for p, _ in result[OK]] == [str(root / "ok.png")]
    assert [p for p, _ in result[TRUNCATED]] == [str(truncated)]
    assert [p for p, _ in result[CORRUPT]] == [str(corrupt)]


def test_diagnose_tree_explains_why(tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "ok.png")
    truncated = root / "trunc.jpg"
    _write_image(truncated, size=(400, 300))
    _truncate(truncated, drop=225)

    result = diagnose_tree(str(root))
    why = dict(result[TRUNCATED])[str(truncated)]
    # the caller can print something better than a bare status
    assert "truncated" in why
    assert "media-match check" in why
    # a clean file gets no explanation, so nothing pointless is printed
    assert dict(result[OK])[str(root / "ok.png")] == ""


def test_repair_truncated_makes_it_loadable(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p, size=(400, 300))
    _truncate(p, drop=225)
    assert check_image(str(p)) == TRUNCATED

    outcome = repair_truncated(str(p))
    assert "repaired" in outcome
    assert check_image(str(p)) == OK  # now loads cleanly
    # original bytes kept as backup
    backup = str(p) + ".truncated"
    assert os.path.exists(backup)
    assert check_image(backup) == TRUNCATED


def test_repair_truncated_keeps_existing_backup(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p, size=(400, 300))
    _truncate(p, drop=225)
    with open(str(p) + ".truncated", "wb") as fh:
        fh.write(b"previous backup")
    outcome = repair_truncated(str(p))
    assert "backup exists" in outcome
    with open(str(p) + ".truncated", "rb") as fh:
        assert fh.read() == b"previous backup"


def test_quarantine_moves_corrupt_file(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    bad = root / "bad.png"
    bad.write_bytes(b"not an image")
    qdir = str(root / ".media-match-quarantine")

    dest = quarantine(str(bad), qdir)
    assert not os.path.exists(str(bad))
    assert os.path.exists(dest)
    assert os.path.dirname(dest) == qdir


def test_quarantine_handles_name_collision(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    qdir = str(root / "q")
    bad = root / "bad.png"
    bad.write_bytes(b"x")
    quarantine(str(bad), qdir)
    bad2 = root / "bad.png"
    bad2.write_bytes(b"y")
    dest = quarantine(str(bad2), qdir)
    assert os.path.basename(dest) == "bad.1.png"
