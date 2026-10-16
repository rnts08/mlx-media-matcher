from media_match import cli
from media_match.db import Database


def _add(db, path, w, h, size):
    db.upsert_file(path=path, size=size, mtime=1.0, sha1=path, width=w, height=h,
                   phash="0" * 16, description="", keywords="")


def _tree(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    db = Database(str(root / ".media-match.db"))
    _add(db, str(root / "big.jpg"), 4000, 3000, 5_000_000)
    _add(db, str(root / "small.jpg"), 200, 150, 20_000)
    db.close()
    return root


def test_duplicates_reports_reclaimable(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["duplicates", str(root)]) == 0
    out = capsys.readouterr().out
    assert "reclaimable" in out
    assert "in 1 extra copy" in out


def test_stats_reports_reclaimable(tmp_path, capsys):
    root = _tree(tmp_path)
    assert cli.main(["stats", str(root)]) == 0
    out = capsys.readouterr().out
    assert "duplicates:" in out
    assert "reclaimable" in out
    assert "1 group(s)" in out
