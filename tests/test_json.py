import json

from media_match import cli
from media_match.db import Database


def _add(db, path, captured="", camera="", w=10, h=10, size=100, sha1=None,
         desc=""):
    db.upsert_file(path=path, size=size, mtime=1.0, sha1=sha1 or path, width=w,
                   height=h, phash="0" * 16, description=desc, keywords="",
                   captured_at=captured, camera=camera)


def _tree(tmp_path):
    root = tmp_path / "tree"
    root.mkdir(exist_ok=True)
    db = Database(str(root / ".media-match.db"))
    _add(db, str(root / "a.jpg"), captured="2021-07-04T12:00:00",
         camera="Apple iPhone 13", desc="a dog")
    _add(db, str(root / "b.jpg"), captured="2020-01-01T00:00:00", desc="a cat")
    db.close()
    return root


def test_search_json(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["search", str(root), "dog", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert data[0]["path"].endswith("a.jpg")
    assert data[0]["kind"] == "image"
    assert data[0]["captured_at"] == "2021-07-04T12:00:00"


def test_tags_json(tmp_path, capsys):
    root = _tree(tmp_path)
    db = Database(str(root / ".media-match.db"))
    fid = db.get_file(str(root / "a.jpg"))["id"]
    db.apply_refinement("image", [(fid, "dog", "dog", "a dog", [("dog", "animal")])])
    db.close()
    assert cli.main(["tags", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"tag": "dog", "count": 1}]


def test_stats_json(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["stats", str(root), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["files"]["total"] == 2
    assert "duplicates" in data and "compute" in data and "dates" in data


def test_clean_dry_run_json(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir(exist_ok=True)
    db = Database(str(root / ".media-match.db"))
    _add(db, str(root / "big.jpg"), w=4000, h=3000, size=5_000_000, sha1="s1")
    _add(db, str(root / "small.jpg"), w=100, h=100, size=1000, sha1="s2")
    db.close()
    assert cli.main(["clean", str(root), "--dry-run", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1
    assert data[0]["path"].endswith("small.jpg")
