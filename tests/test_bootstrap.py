"""The bootstrap command: walk the tree, record filename/type/size, and
with flags prune duplicates / non-media files and consolidate extensions."""

import os

import pytest
from PIL import Image

from media_match import cli
from media_match.bootstrap import (
    classify_extension, consolidated_path, detect_media_type,
    find_exact_duplicates, fixed_extension_path, walk_tree,
)
from media_match.db import Database


def _write_image(path, size=(64, 48), color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


# -- extension classification & consolidation --------------------------------


def test_classify_extension():
    assert classify_extension(".jpg") == "image"
    assert classify_extension(".mp4") == "video"
    assert classify_extension(".mp3") == "audio"
    assert classify_extension(".zip") == "other"
    assert classify_extension("") == "other"


def test_consolidated_path_normalises_case_and_alias():
    assert consolidated_path("/a/photo.JPEG") == "/a/photo.jpg"
    assert consolidated_path("/a/photo.jpeg") == "/a/photo.jpg"
    assert consolidated_path("/a/clip.MP4") == "/a/clip.mp4"
    assert consolidated_path("/a/photo.jpg") is None  # already canonical
    assert consolidated_path("/a/archive.zip") is None  # not media


def test_detect_media_type_jpeg(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p)
    assert detect_media_type(str(p)) == ".jpg"


def test_fixed_extension_path_adds_missing(tmp_path):
    src = tmp_path / "real.jpg"
    _write_image(src)
    p = tmp_path / "IMG_1234"
    p.write_bytes(src.read_bytes())
    new = fixed_extension_path(str(p))
    assert new == str(p) + ".jpg"


def test_fixed_extension_path_leaves_known_alone(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p)
    assert fixed_extension_path(str(p)) is None


# -- tree walking --------------------------------------------------------------


def test_walk_tree_records_all_kinds(tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    (root / "notes.txt").write_text("hello")
    (root / "data.zip").write_bytes(b"PK\x03\x04")
    rows = walk_tree(str(root))
    kinds = {os.path.basename(r[0]): r[4] for r in rows}
    assert kinds["a.jpg"] == "image"
    assert kinds["notes.txt"] == "other"
    assert kinds["data.zip"] == "other"


def test_walk_tree_empty_dir(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert walk_tree(str(root)) == []


# -- duplicate finding ---------------------------------------------------------


def test_find_exact_duplicates_groups_identical_files(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.jpg")
    _write_image(root / "copy.jpg")  # same content
    _write_image(root / "different.jpg", color=(10, 20, 30))
    rows = walk_tree(str(root))
    groups = find_exact_duplicates(rows)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_find_exact_duplicates_no_groups_when_all_different(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.jpg", color=(1, 2, 3))
    _write_image(root / "b.jpg", color=(4, 5, 6))
    rows = walk_tree(str(root))
    assert find_exact_duplicates(rows) == []


# -- Database bootstrap table ---------------------------------------------------


def test_bootstrap_table_round_trip(tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    db_file = str(tmp_path / "t.db")
    rows = walk_tree(str(root))
    db = Database(db_file)
    db.replace_bootstrap(rows)
    assert db.bootstrap_counts() == {"image": 1}
    stored = {r["path"]: r for r in db.all_bootstrap()}
    assert str(root / "a.jpg") in stored
    db.close()


def test_bootstrap_replace_clears_old_rows(tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    db_file = str(tmp_path / "t.db")
    db = Database(db_file)
    db.replace_bootstrap(walk_tree(str(root)))
    db.replace_bootstrap([])  # second pass clears
    assert db.all_bootstrap() == []
    db.close()


# -- CLI -----------------------------------------------------------------------


def test_bootstrap_dry_run_builds_index_without_touching(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    (root / "notes.txt").write_text("hello")
    assert cli.main(["bootstrap", str(root)]) == 0
    out = capsys.readouterr().out
    assert "2 file(s)" in out
    assert "1 image(s)" in out
    assert "1 other" in out
    # files are not touched in dry-run
    assert (root / "a.jpg").exists()
    assert (root / "notes.txt").exists()


def test_bootstrap_media_only_consolidates_and_prunes(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "photo.JPEG")
    (root / "archive.zip").write_bytes(b"PK\x03\x04")
    assert cli.main(["bootstrap", str(root), "--media-only", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "pruned 1 non-media" in out
    assert "renamed 1 file(s)" in out
    # extension consolidated, non-media pruned
    assert (root / "photo.jpg").exists()
    assert not (root / "photo.JPEG").exists()
    assert not (root / "archive.zip").exists()


def test_bootstrap_media_only_dry_run_shows_plan(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "photo.JPEG")
    (root / "archive.zip").write_bytes(b"PK\x03\x04")
    assert cli.main(["bootstrap", str(root), "--media-only"]) == 0
    out = capsys.readouterr().out
    assert "1 non-media file(s) to prune" in out
    assert "1 file(s) to consolidate" in out
    # dry-run: nothing touched
    assert (root / "photo.JPEG").exists()
    assert (root / "archive.zip").exists()


def test_bootstrap_prune_keeps_newest(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "old.jpg")
    _write_image(root / "new.jpg")  # same content
    # make new.jpg newer
    import time

    now = time.time()
    os.utime(root / "old.jpg", (now - 100, now - 100))
    os.utime(root / "new.jpg", (now, now))
    assert cli.main(["bootstrap", str(root), "--prune", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "trashed 1 duplicate" in out
    assert (root / "new.jpg").exists()


def test_bootstrap_fix_extensions(capsys, tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    # a JPEG with no extension
    src = tmp_path / "real.jpg"
    _write_image(src)
    (root / "IMG_1234").write_bytes(src.read_bytes())
    assert cli.main(["bootstrap", str(root), "--media-only",
                     "--fix-extensions", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "1 file(s) to fix missing extensions" in out
    assert (root / "IMG_1234.jpg").exists()
