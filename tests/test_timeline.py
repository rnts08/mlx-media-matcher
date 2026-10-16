from media_match import cli
from media_match.db import Database


def _add(db, path, captured, camera="", w=10, h=10, size=100, sha1=None):
    db.upsert_file(path=path, size=size, mtime=1.0, sha1=sha1 or path, width=w,
                   height=h, phash="0" * 16, description="", keywords="",
                   captured_at=captured, camera=camera)


def _tree(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    db = Database(str(root / ".media-match.db"))
    _add(db, str(root / "a.jpg"), "2019-05-01T10:00:00", camera="Canon EOS 5D")
    _add(db, str(root / "b.jpg"), "2021-07-04T12:00:00", camera="Apple iPhone 13")
    _add(db, str(root / "c.jpg"), "2023-01-15T09:00:00", camera="Canon EOS 5D")
    db.close()
    return root


def test_parse_day_boundaries():
    assert cli._parse_day("2021", end=False) == "2021-01-01"
    assert cli._parse_day("2021", end=True) == "2021-12-31T23:59:59"
    assert cli._parse_day("2021-07", end=False) == "2021-07-01"
    assert cli._parse_day("2021-07", end=True) == "2021-07-31T23:59:59"
    assert cli._parse_day("2021-07-04", end=False) == "2021-07-04"
    assert cli._parse_day("2021-07-04", end=True) == "2021-07-04T23:59:59"
    assert cli._parse_day("nonsense", end=False) is None
    assert cli._parse_day("", end=False) is None


def test_search_since_filters(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["search", str(root), "--since", "2021-01-01"]) == 0
    out = capsys.readouterr().out
    assert "b.jpg" in out and "c.jpg" in out
    assert "a.jpg" not in out


def test_search_until_filters(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["search", str(root), "--until", "2020-12-31"]) == 0
    out = capsys.readouterr().out
    assert "a.jpg" in out
    assert "b.jpg" not in out and "c.jpg" not in out


def test_search_camera_filters(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["search", str(root), "--camera", "Canon"]) == 0
    out = capsys.readouterr().out
    assert "a.jpg" in out and "c.jpg" in out
    assert "b.jpg" not in out


def test_search_bad_date_errors(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["search", str(root), "--since", "not-a-date"]) == 2
    assert "expected a date" in capsys.readouterr().out


def test_duplicates_prefer_date_picks_newest(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    db = Database(str(root / ".media-match.db"))
    _add(db, str(root / "orig.jpg"), "2019-01-01T00:00:00",
         w=4000, h=3000, size=5_000_000, sha1="s1")
    _add(db, str(root / "copy.jpg"), "2022-01-01T00:00:00",
         w=100, h=100, size=1000, sha1="s2")
    db.close()

    assert cli.main(["duplicates", str(root), "--prefer", "date"]) == 0
    best = next(l for l in capsys.readouterr().out.splitlines() if "[best]" in l)
    assert "copy.jpg" in best
