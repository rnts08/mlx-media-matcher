"""CLI commands for index metadata: clear / import / export / reset.

The metadata commands are the safety net around the VLM captions: export is
the backup, clear wipes description/keywords/type/tags, import restores them.
'reset' deletes the index database itself (media files are never touched).
"""

import json
import os

import numpy as np
from PIL import Image

from media_match import cli
from media_match.db import Database
from media_match.pipeline import build_jobs, run_scan, store_result


class _FakeEncoder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_images(self, arrays):
        return np.stack([np.ones(self.dim, dtype=np.float32) for _ in arrays])


def _write_image(path, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 30, 200)).save(path)


def _indexed(tmp_path, name="a.png"):
    """A tree with one indexed, described image; returns (root, db path)."""
    root = tmp_path / "tree"
    _write_image(root / name)
    db_file = str(tmp_path / "t.db")
    db = Database(db_file)
    jobs, *_ = build_jobs(str(root), db, include_video=False, reanalyze=False)
    collected = []
    run_scan(jobs, _FakeEncoder(), interval=1.0, max_frames=120,
             batch_size=4, workers=2, on_result=collected.append)
    for r in collected:
        store_result(db, r)
    db.set_description(db.all_files()[0].id, "a purple square", "purple, square")
    db.close()
    return root, db_file


# -- export-metadata ----------------------------------------------------------


def test_export_metadata_round_trips(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    out = tmp_path / "meta.json"
    assert cli.main(["export-metadata", str(root), "--db", db_file,
                     "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["kind"] == "image"
    assert data[0]["path"] == "a.png"
    assert data[0]["description"] == "a purple square"
    assert data[0]["keywords"] == "purple, square"
    assert data[0]["type"] == ""


# -- clear-metadata -----------------------------------------------------------


def test_clear_metadata_one_item(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    assert cli.main(["clear-metadata", str(root), "--db", db_file,
                     "--yes", "a.png"]) == 0
    db = Database(db_file)
    rec = db.all_files()[0]
    assert rec.description == "" and rec.keywords == ""
    assert rec.raw_keywords == ""  # refine must not resurrect the old keywords
    assert db.has_tags() == 0
    db.close()


def test_clear_metadata_unknown_path_is_reported(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    assert cli.main(["clear-metadata", str(root), "--db", db_file,
                     "--yes", "nope.jpg"]) == 0
    assert "not in index" in capsys.readouterr().out


def test_clear_metadata_all_kinds(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    assert cli.main(["clear-metadata", str(root), "--db", db_file,
                     "--yes"]) == 0
    db = Database(db_file)
    assert db.all_files()[0].description == ""
    db.close()


# -- import-metadata ----------------------------------------------------------


def test_import_metadata_restores_export(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    out = tmp_path / "meta.json"
    cli.main(["export-metadata", str(root), "--db", db_file, "--out", str(out)])

    assert cli.main(["clear-metadata", str(root), "--db", db_file, "--yes"]) == 0
    db = Database(db_file)
    assert db.all_files()[0].description == ""
    db.close()

    assert cli.main(["import-metadata", "--db", db_file, str(root),
                     str(out)]) == 0
    db = Database(db_file)
    rec = db.all_files()[0]
    assert rec.description == "a purple square"
    assert rec.keywords == "purple, square"
    assert rec.type_score == 1.0
    db.close()

    # ... and FTS finds it again
    assert cli.main(["search", str(root), "--db", db_file, "--fts",
                     "purple"]) == 0
    assert "a.png" in capsys.readouterr().out


def test_import_metadata_csv(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    csv_file = tmp_path / "meta.csv"
    csv_file.write_text(
        "path,description,keywords\n"
        "missing.jpg,ignored,x\n"
        "a.png,a caption,cap\n",
        encoding="utf-8",
    )
    assert cli.main(["import-metadata", "--db", db_file, str(root),
                     str(csv_file)]) == 0
    out = capsys.readouterr().out
    assert "not in index: missing.jpg" in out
    db = Database(db_file)
    assert db.all_files()[0].description == "a caption"
    db.close()


def test_import_metadata_missing_file(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    assert cli.main(["import-metadata", "--db", db_file, str(root),
                     str(tmp_path / "nope.json")]) == 2


# -- reset --------------------------------------------------------------------


def test_reset_removes_index_only(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    media = root / "a.png"
    assert media.exists()

    assert cli.main(["reset", str(root), "--db", db_file, "--yes"]) == 0
    assert "index reset" in capsys.readouterr().out
    assert not os.path.exists(db_file)
    assert media.exists()  # media files are never touched


def test_reset_without_index_is_a_noop(capsys, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert cli.main(["reset", str(root), "--yes"]) == 0
    assert "nothing to reset" in capsys.readouterr().out


def test_reset_removes_sidecars(capsys, tmp_path):
    root, db_file = _indexed(tmp_path)
    for suffix in ("-wal", "-shm"):
        with open(db_file + suffix, "wb") as fh:
            fh.write(b"")
    assert cli.main(["reset", str(root), "--db", db_file, "--yes"]) == 0
    assert not os.path.exists(db_file + "-wal")
    assert not os.path.exists(db_file + "-shm")
