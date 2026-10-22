"""The standalone `embed` command's model-space guard."""

from media_match import cli
from media_match.db import Database


def _add(db, path, *, embedding, sha1):
    db.upsert_file(
        path=str(path), size=10, mtime=1.0, sha1=sha1, width=100, height=80,
        phash="0" * 16, description="", keywords="", embedding=embedding,
    )


def test_embed_refuses_to_mix_vector_spaces_without_recompute(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    db = Database(str(root / ".media-match.db"))
    db.set_meta("clip_model", "old/clip")
    _add(db, root / "embedded.jpg", embedding=b"\0" * 8, sha1="a1")
    _add(db, root / "missing.jpg", embedding=None, sha1="a2")
    db.close()

    assert cli.main(["embed", str(root), "--clip-model", "new/clip"]) == 2
    err = capsys.readouterr().err
    assert "old/clip" in err and "new/clip" in err
    assert "--recompute" in err
