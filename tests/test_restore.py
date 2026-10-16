import os

from media_match import cli
from media_match.cleanup import restore_file
from media_match.db import Database


def test_restore_file_roundtrip(tmp_path, monkeypatch):
    trash = tmp_path / "Trash"
    trash.mkdir()
    original = tmp_path / "Pictures" / "photo.jpg"
    original.parent.mkdir()
    original.write_bytes(b"photo")

    trashed = trash / "photo.jpg"
    os.rename(original, trashed)
    monkeypatch.setattr("media_match.cleanup.trash_dirs", lambda: [str(trash)])

    assert not original.exists()
    assert restore_file(str(original)) == "restored"
    assert original.exists()
    assert not trashed.exists()

    assert restore_file(str(original)) == "already"

    os.rename(original, trashed)
    os.rename(trashed, trash / "other.jpg")
    assert restore_file(str(original)) == "missing"


def test_record_and_clear_trash_log(tmp_path):
    db = Database(str(tmp_path / "t.db"))

    class R:
        path = "/x/a.jpg"
        size = 100

    db.record_trash([R()], kind="image", batch="20261017T120000")
    rows = db.trash_log()
    assert len(rows) == 1
    assert rows[0]["path"] == "/x/a.jpg"
    assert rows[0]["batch"] == "20261017T120000"

    db.clear_trash_log([rows[0]["id"]])
    assert db.trash_log() == []
    db.close()


def test_cmd_restore_dry_run(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    db = Database(str(root / ".media-match.db"))

    class R:
        path = str(root / "a.jpg")
        size = 100

    db.record_trash([R()], kind="image", batch="b1")
    db.close()

    assert cli.main(["restore", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "a.jpg" in out
    assert "Dry run" in out
