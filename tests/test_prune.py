from pathlib import Path

from media_match import cli
from media_match.db import Database
from media_match.pipeline import prune_missing


def _seed(root: Path) -> Database:
    root.mkdir(parents=True, exist_ok=True)
    db = Database(str(root / ".media-match.db"))
    for name in ("a.jpg", "b.jpg"):
        path = root / name
        path.write_bytes(b"x")
        db.upsert_file(path=str(path), size=1, mtime=1.0, sha1=name, width=1,
                       height=1, phash="0" * 16, description="", keywords="")
    video = root / "clip.mp4"
    video.write_bytes(b"x")
    db.upsert_video(path=str(video), size=1, mtime=1.0, sha1="v", duration=1.0,
                    width=1, height=1, fps=1.0, codec="h264", bitrate=0,
                    signature="0" * 16)
    return db


def test_prune_removes_missing_media(tmp_path):
    db = _seed(tmp_path)
    (tmp_path / "a.jpg").unlink()
    (tmp_path / "clip.mp4").unlink()

    # Images-only scan leaves the stale video row alone.
    assert prune_missing(str(tmp_path), db, include_video=False) == 1
    assert db.count_files() == 1
    assert db.count_videos() == 1

    # Including videos removes the ghost video too.
    assert prune_missing(str(tmp_path), db, include_video=True) == 1
    assert db.count_videos() == 0
    db.close()


def test_prune_scoped_to_root_and_skips_missing_root(tmp_path):
    db = _seed(tmp_path / "tree")
    outside = tmp_path / "elsewhere.jpg"
    db.upsert_file(path=str(outside), size=1, mtime=1.0, sha1="o", width=1,
                   height=1, phash="0" * 16, description="", keywords="")

    # A row outside the scanned root is never touched.
    assert prune_missing(str(tmp_path / "tree"), db, include_video=True) == 0
    assert db.count_files() == 3

    # A root that does not exist (unmounted drive) must not wipe the index.
    assert prune_missing(str(tmp_path / "gone"), db, include_video=True) == 0
    assert db.count_files() == 3
    db.close()


def test_scan_prune_flag_reports_and_removes(tmp_path, capsys):
    db = _seed(tmp_path)
    for name in ("a.jpg", "b.jpg", "clip.mp4"):
        (tmp_path / name).unlink()
    db.close()

    assert cli.main(["scan", str(tmp_path), "--prune"]) == 0
    out = capsys.readouterr().out
    assert "stale index row" in out

    reopened = Database(str(tmp_path / ".media-match.db"))
    assert reopened.count_files() == 0
    reopened.close()
