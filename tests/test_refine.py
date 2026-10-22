import numpy as np

from media_match import cli
from media_match.db import Database
from media_match.groups import cluster_by_similarity


def _vec(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _add(db, path, vec, keywords, sha1, description="a test photo"):
    db.upsert_file(
        path=path, size=10, mtime=1.0, sha1=sha1, width=100, height=80,
        phash="0" * 16, description=description, keywords=keywords,
        embedding=vec.tobytes(),
    )


def _seed(root):
    db = Database(str(root / ".media-match.db"))
    # two near-identical "Alice" photos and one unrelated "Bob"
    _add(db, str(root / "alice1.jpg"), _vec(1.0, 0.0, 0.0, 0.0), "women, a jacket", "a1")
    _add(db, str(root / "alice2.jpg"), _vec(0.99, 0.05, 0.0, 0.0), "woman, hat", "a2")
    _add(db, str(root / "bob.jpg"), _vec(0.0, 0.0, 1.0, 0.0), "man, bicycle", "b1")
    db.close()


def test_refine_writes_tags_and_facets(tmp_path):
    _seed(tmp_path)
    assert cli.main(["refine", str(tmp_path)]) == 0

    db = Database(str(tmp_path / ".media-match.db"))
    tags = dict(db.tag_counts(kind="image"))
    assert tags["woman"] == 2 and tags["man"] == 1 and tags["bicycle"] == 1

    grouped = db.facet_tag_counts("image")
    assert ("woman", 2) in grouped["person"]
    assert ("bicycle", 1) in grouped["object"]

    person_ids = db.media_ids_with_facet("person")
    assert len(person_ids) == 3

    # the original keywords are preserved
    row = db.conn.execute(
        "SELECT raw_keywords, keywords FROM files WHERE path = ?",
        (str(tmp_path / "alice1.jpg"),),
    ).fetchone()
    assert "women, a jacket" in row["raw_keywords"]
    assert row["keywords"] == "woman, jacket"
    db.close()


def test_people_clusters_and_labels(tmp_path, capsys):
    _seed(tmp_path)
    assert cli.main(["refine", str(tmp_path)]) == 0
    capsys.readouterr()

    assert cli.main(["people", str(tmp_path), "--threshold", "0.9"]) == 0
    out = capsys.readouterr().out
    assert "Person 1" in out and "(2 photos)" in out
    assert "Person 2" in out and "(1 photos)" in out

    # name the two-photo cluster, then list it by name
    assert cli.main(["people", str(tmp_path), "--label", "1=Alice"]) == 0
    capsys.readouterr()
    assert cli.main(["search", str(tmp_path), "--cluster", "Alice"]) == 0
    out = capsys.readouterr().out
    assert "alice1.jpg" in out and "alice2.jpg" in out
    assert "bob.jpg" not in out


def test_people_rerun_keeps_labels(tmp_path, capsys):
    _seed(tmp_path)
    cli.main(["refine", str(tmp_path)])
    cli.main(["people", str(tmp_path), "--threshold", "0.9"])
    cli.main(["people", str(tmp_path), "--label", "1=Alice"])
    capsys.readouterr()

    # re-clustering with the same members must carry the name over
    assert cli.main(["people", str(tmp_path), "--threshold", "0.9"]) == 0
    out = capsys.readouterr().out
    assert "[Alice]" in out


def test_scan_can_refine_tags(tmp_path, capsys):
    # nothing on disk to analyze, but --refine still rebuilds tags from the
    # injected keywords without loading CLIP
    _seed(tmp_path)
    assert cli.main(["scan", str(tmp_path), "--refine"]) == 0
    capsys.readouterr()
    db = Database(str(tmp_path / ".media-match.db"))
    assert dict(db.tag_counts(kind="image"))["woman"] == 2
    assert db.media_ids_with_facet("person")
    db.close()


def test_cluster_by_similarity_keeps_singletons():
    matrix = np.stack([
        _vec(1.0, 0.0, 0.0),
        _vec(0.99, 0.1, 0.0),
        _vec(0.0, 0.0, 1.0),
    ])
    labels = cluster_by_similarity(matrix, threshold=0.9)
    assert labels[0] == labels[1]
    assert labels[2] not in (labels[0],)
    assert len(set(labels)) == 2


def test_refine_requires_descriptions(tmp_path, capsys):
    """Refinement needs captions; an undescribed tree must say so, not no-op."""
    root = tmp_path / "tree"
    root.mkdir()
    db = Database(str(root / ".media-match.db"))
    _add(db, str(root / "a.jpg"), _vec(1.0, 0.0), "woman, jacket", "a1",
         description="")
    db.close()

    assert cli.main(["refine", str(root)]) == 1
    err = capsys.readouterr().err
    assert "No descriptions" in err
    assert "media-match describe" in err
