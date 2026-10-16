import media_match.config as cfgmod
from media_match.config import is_ignored, load_config, load_ignore
from media_match.db import Database
from media_match.pipeline import iter_media, prune_missing


def test_load_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(tmp_path / "nope.toml"))
    assert load_config() == {}


def test_load_config_parses(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[media-match]\nworkers = 8\nthreshold = 4\n')
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(path))
    assert load_config() == {"workers": 8, "threshold": 4}


def test_load_ignore(tmp_path):
    (tmp_path / ".media-matchignore").write_text("# comment\nnode_modules/\n*.tmp\n")
    assert load_ignore(str(tmp_path)) == ["node_modules/", "*.tmp"]


def test_is_ignored():
    assert is_ignored("node_modules/foo.js", ["node_modules/"])
    assert is_ignored("a/b.tmp", ["*.tmp"])
    assert not is_ignored("a/b.jpg", ["*.tmp"])


def test_iter_media_respects_ignore(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "keep.jpg").write_bytes(b"x")
    (root / "skip.tmp").write_bytes(b"x")
    nm = root / "node_modules"
    nm.mkdir()
    (nm / "x.jpg").write_bytes(b"x")
    found = {p for _k, p in iter_media(
        str(root), include_video=False, ignore=["node_modules/", "*.tmp"])}
    assert found == {str(root / "keep.jpg")}


def test_prune_missing_drops_ignored(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "keep.jpg").write_bytes(b"x")
    (root / "skip.tmp").write_bytes(b"x")
    db = Database(str(tmp_path / "t.db"))
    for name in ("keep.jpg", "skip.tmp"):
        db.upsert_file(path=str(root / name), size=1, mtime=1.0, sha1=name,
                       width=1, height=1, phash="0" * 16, description="",
                       keywords="")
    removed = prune_missing(str(root), db, include_video=False, ignore=["*.tmp"])
    assert removed == 1
    assert db.count_files() == 1
    db.close()
