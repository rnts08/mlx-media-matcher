import sqlite3

from media_match.db import SCHEMA_VERSION, Database
from media_match.pipeline import iter_media


def test_schema_version_is_stamped(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION)
    db.close()


def test_pre_versioning_db_is_migrated(tmp_path):
    path = str(tmp_path / "old.db")
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, "
        "size INTEGER, mtime REAL, sha1 TEXT, width INTEGER, height INTEGER, "
        "phash TEXT, description TEXT DEFAULT '', keywords TEXT DEFAULT '', "
        "scanned_at TEXT DEFAULT '')"
    )
    con.execute(
        "CREATE TABLE videos (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, "
        "size INTEGER, mtime REAL, sha1 TEXT, duration REAL, width INTEGER, "
        "height INTEGER, fps REAL, codec TEXT, bitrate INTEGER, frame_count INTEGER, "
        "signature TEXT, description TEXT, keywords TEXT, scanned_at TEXT)"
    )
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    con.commit()
    con.close()

    db = Database(path)
    file_cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(files)")}
    video_cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(videos)")}
    assert {"embedding", "raw_keywords", "type", "type_score", "cluster_id"} <= file_cols
    assert "raw_keywords" in video_cols
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION)
    db.close()


def _add_file(db, path, sha1="a"):
    db.upsert_file(path=path, size=1, mtime=1.0, sha1=sha1, width=1, height=1,
                   phash="0" * 16, description="", keywords="")
    return db.get_file(path)["id"]


def test_delete_files_clears_tags_and_clusters(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    fid = _add_file(db, "/x/a.jpg")
    db.apply_refinement("image", [(fid, "dog", "dog", "", [("dog", "animal")])])
    db.set_clusters("image", {1: [fid]})
    assert db.has_tags("image") == 1
    assert len(db.all_clusters("image")) == 1

    db.delete_files([fid])

    assert db.has_tags("image") == 0
    assert db.all_clusters("image") == []
    db.close()


def test_delete_videos_clears_tags(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    vid = db.upsert_video(path="/x/a.mp4", size=1, mtime=1.0, sha1="v",
                          duration=1.0, width=1, height=1, fps=1.0, codec="h264",
                          bitrate=1, signature="")
    db.apply_refinement("video", [(vid, "dog", "dog", "", [("dog", "animal")])])
    assert db.has_tags("video") == 1

    db.delete_videos([vid])

    assert db.has_tags("video") == 0
    db.close()


def test_record_scan_uses_utc(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_scan(items=1, frames=1, bytes_read=1, wall=1.0, cpu=1.0, gpu=1.0)
    assert db.scan_summary()["last_scan_at"].endswith("+00:00")
    db.close()


def test_iter_media_skips_dotfiles_and_dotdirs(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "real.jpg").write_bytes(b"x")
    (root / ".trashed-123.jpg").write_bytes(b"x")
    (root / "._apple.jpg").write_bytes(b"x")
    hidden = root / ".git"
    hidden.mkdir()
    (hidden / "pic.png").write_bytes(b"x")

    found = {p for _kind, p in iter_media(str(root), include_video=False)}
    assert found == {str(root / "real.jpg")}
