"""The standalone `classify` command."""

from media_match import cli
from media_match.db import Database


def _db(root):
    return Database(str(root / ".media-match.db"))


def _add(db, path, *, embedding=b"\0" * 8, type_name="", sha1="a1"):
    db.upsert_file(
        path=str(path), size=10, mtime=1.0, sha1=sha1, width=100, height=80,
        phash="0" * 16, description="a photo", keywords="photo",
        embedding=embedding,
    )
    if type_name:
        row = db.get_file(str(path))
        db.set_types([(row["id"], type_name, 1.0)])


def test_classify_without_an_index(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    assert cli.main(["classify", str(root)]) == 1
    assert "No index" in capsys.readouterr().err


def test_classify_empty_index(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    _db(root).close()
    assert cli.main(["classify", str(root)]) == 0
    assert "No indexed images" in capsys.readouterr().out


def test_classify_explains_missing_embeddings(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    db = _db(root)
    _add(db, root / "a.jpg", embedding=None)
    db.close()

    assert cli.main(["classify", str(root)]) == 1
    err = capsys.readouterr().err
    assert "No CLIP embeddings" in err
    assert "scan" in err and "embed" in err


def test_classify_says_when_everything_is_typed(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    db = _db(root)
    _add(db, root / "a.jpg", type_name="selfie")
    db.close()

    assert cli.main(["classify", str(root)]) == 0
    out = capsys.readouterr().out
    assert "already have a type" in out
    assert "--recompute" in out


def test_classify_mentions_images_left_without_embeddings(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    db = _db(root)
    _add(db, root / "typed.jpg", type_name="selfie", sha1="a1")
    _add(db, root / "unembedded.jpg", embedding=None, sha1="a2")
    db.close()

    assert cli.main(["classify", str(root)]) == 0
    out = capsys.readouterr().out
    assert "already have a type" in out
    assert "1 image(s) have no embedding" in out
    assert "media-match embed" in out
