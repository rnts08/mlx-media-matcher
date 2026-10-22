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
    assert consolidated_path("/a/song.oga") == "/a/song.ogg"
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
    (root / "song.mp3").write_bytes(b"ID3\x04\x00\x00")
    rows = walk_tree(str(root))
    kinds = {os.path.basename(r[0]): r[4] for r in rows}
    assert kinds["a.jpg"] == "image"
    assert kinds["song.mp3"] == "audio"
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


def test_bootstrap_media_only_keeps_audio(tmp_path, capsys):
    # Audio is media too: --media-only may prune notes and archives, but must
    # not treat a music file as junk before audio analysis exists.
    root = tmp_path / "tree"
    root.mkdir()
    (root / "song.mp3").write_bytes(b"ID3\x04\x00\x00")
    (root / "notes.txt").write_text("hello")
    (root / "archive.zip").write_bytes(b"PK\x03\x04")

    assert cli.main(["bootstrap", str(root), "--media-only", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "1 audio" in out
    assert "pruned 2 non-media" in out
    assert (root / "song.mp3").exists()
    assert not (root / "notes.txt").exists()
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

    db = Database(str(root / ".media-match.db"))
    paths = {r["path"] for r in db.all_bootstrap()}
    db.close()
    assert str(root / "new.jpg") in paths
    assert str(root / "old.jpg") not in paths


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


# -- --move-to: move non-media files instead of trashing them ------------------


def test_move_to_moves_instead_of_trashing(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text("hello")
    dest = tmp_path / "elsewhere"

    assert cli.main(["bootstrap", str(root), "--media-only",
                     "--move-to", str(dest), "--yes"]) == 0
    out = capsys.readouterr().out
    assert "moved 1 non-media file(s)" in out
    assert "pruned" not in out
    # media stayed put; the non-media file kept its layout under dest
    assert (root / "a.jpg").exists()
    assert not (root / "docs" / "notes.txt").exists()
    assert (dest / "docs" / "notes.txt").read_text() == "hello"


def test_move_to_dry_run_changes_nothing(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    (root / "notes.txt").write_text("hello")
    dest = tmp_path / "elsewhere"

    assert cli.main(["bootstrap", str(root), "--media-only",
                     "--move-to", str(dest)]) == 0
    out = capsys.readouterr().out
    assert f"1 non-media file(s) to move to {dest}" in out
    assert (root / "notes.txt").exists()
    assert not dest.exists()


def test_move_to_destination_inside_tree_is_not_rescanned(capsys, tmp_path):
    # A destination under the root must not be walked, or the moved files
    # would be found and moved again on the next run.
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    (root / "notes.txt").write_text("hello")
    dest = root / "_nonmedia"

    assert cli.main(["bootstrap", str(root), "--media-only",
                     "--move-to", str(dest), "--yes"]) == 0
    assert (dest / "notes.txt").exists()

    assert cli.main(["bootstrap", str(root), "--media-only",
                     "--move-to", str(dest), "--yes"]) == 0
    assert "0 other" in capsys.readouterr().out
    assert (dest / "notes.txt").read_text() == "hello"
    assert not (dest / "notes (2).txt").exists()


def test_move_to_avoids_overwriting_same_named_files(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    (root / "notes.txt").write_text("from tree")
    dest = tmp_path / "elsewhere"
    dest.mkdir()
    (dest / "notes.txt").write_text("already there")

    assert cli.main(["bootstrap", str(root), "--media-only",
                     "--move-to", str(dest), "--yes"]) == 0
    # the pre-existing file is untouched, the new one is renamed aside
    assert (dest / "notes.txt").read_text() == "already there"
    assert (dest / "notes (2).txt").read_text() == "from tree"


def test_move_to_requires_media_only(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    assert cli.main(["bootstrap", str(root), "--move-to",
                     str(tmp_path / "elsewhere")]) == 2
    assert "--move-to only applies to --media-only" in capsys.readouterr().err


def test_move_to_rejects_the_tree_itself(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    assert cli.main(["bootstrap", str(root), "--media-only",
                     "--move-to", str(root)]) == 2
    assert "other than the tree itself" in capsys.readouterr().err


# -- help ----------------------------------------------------------------------


def test_command_help_word_does_not_open_a_database(capsys, tmp_path):
    # 'bootstrap help' used to treat 'help' as the root path and crash while
    # opening a database inside a folder that does not exist.
    assert cli.main(["bootstrap", "help"]) == 0
    assert "bootstrap —" in capsys.readouterr().out


def test_every_listed_command_has_help(capsys):
    for name in cli._COMMAND_SUMMARIES:
        if name == "help":
            continue
        assert cli.main(["help", name]) == 0, name
        assert f"{name} —" in capsys.readouterr().out, name


def test_command_help_word_is_ignored_as_an_option_value(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "a.jpg")
    (root / "notes.txt").write_text("hello")
    # 'help' here is the value of --db, not a request for help
    assert cli.main(["bootstrap", str(root), "--db",
                     str(tmp_path / "help"), "--media-only", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "1 non-media file(s) to prune" in out
    assert (tmp_path / "help").exists()


def test_bootstrap_checks_loadability_by_default(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "ok.jpg")
    (root / "broken.jpg").write_bytes(b"not an image")
    assert cli.main(["bootstrap", str(root)]) == 0
    out = capsys.readouterr().out
    assert "1 ok" in out and "1 corrupt" in out
    assert "broken.jpg" in out
    assert "Quarantine 1 corrupt file(s)?" in out
    # a non-terminal means the repair prompt is skipped; the file stays put
    assert (root / "broken.jpg").exists()


def test_bootstrap_no_check_skips_the_loadability_pass(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "ok.jpg")
    (root / "broken.jpg").write_bytes(b"not an image")
    assert cli.main(["bootstrap", str(root), "--no-check"]) == 0
    out = capsys.readouterr().out
    assert "corrupt" not in out
    assert "Quarantine" not in out


def test_bootstrap_yes_applies_check_repairs_and_refreshes_index(capsys, tmp_path):
    root = tmp_path / "tree"
    _write_image(root / "ok.jpg")
    bad = root / "broken.jpg"
    bad.write_bytes(b"not an image")
    assert cli.main(["bootstrap", str(root), "--yes"]) == 0
    assert not bad.exists()
    quarantined = root / ".media-match-quarantine" / "broken.jpg"
    assert quarantined.exists()

    from media_match.db import Database

    db = Database(str(root / ".media-match.db"))
    paths = {r["path"] for r in db.all_bootstrap()}
    db.close()
    assert str(root / "ok.jpg") in paths
    assert str(bad) not in paths
