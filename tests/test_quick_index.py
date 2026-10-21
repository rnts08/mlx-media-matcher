"""Tests for `scan --quick` (basic rows) and the inclusive id window.

The quick pass must never read file contents, never load CLIP, and never
downgrade an analysed row. The id window selects indexed rows on both tables
independently and must be honoured by every media-row command.
"""

import json
import os

import pytest
from PIL import Image

from media_match import cli
from media_match.db import Database
from media_match.groups import find_duplicate_groups
from media_match.pipeline import build_jobs
from media_match.video_groups import find_video_duplicate_groups


def _write_image(path, size=(64, 48), color=(120, 30, 200)):
    Image.new("RGB", size, color).save(path)


def _tree(tmp_path, n=3):
    root = tmp_path / "tree"
    root.mkdir()
    for i in range(n):
        _write_image(root / f"{i}.png", color=(i * 40, 10, 10))
    return root


def _rows(db, table="files"):
    return [dict(r) for r in db.conn.execute(f"SELECT * FROM {table} ORDER BY id")]


# -- scan --quick ------------------------------------------------------------


def test_quick_scan_creates_basic_rows_without_analysis(tmp_path):
    root = _tree(tmp_path)
    assert cli.main(["scan", str(root), "--quick"]) == 0
    db = Database(str(root / ".media-match.db"))
    rows = _rows(db)
    db.close()
    assert len(rows) == 3
    for r in rows:
        assert r["sha1"] == "" and r["phash"] == ""
        assert r["width"] == 0 and r["height"] == 0
        assert r["embedding"] is None
        assert r["size"] > 0


def test_quick_scan_videos_need_no_ffprobe(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"not really a video")
    assert cli.main(["scan", str(root), "--quick", "--video"]) == 0
    db = Database(str(root / ".media-match.db"))
    videos = _rows(db, "videos")
    db.close()
    assert len(videos) == 1
    assert videos[0]["sha1"] == "" and videos[0]["duration"] == 0


def test_quick_scan_rerun_adds_nothing_and_limit_advances(tmp_path):
    root = _tree(tmp_path, n=5)
    assert cli.main(["scan", str(root), "--quick", "--limit", "2"]) == 0
    db = Database(str(root / ".media-match.db"))
    assert db.count_files() == 2
    db.close()
    assert cli.main(["scan", str(root), "--quick", "--limit", "2"]) == 0
    db = Database(str(root / ".media-match.db"))
    assert db.count_files() == 4
    db.close()
    assert cli.main(["scan", str(root), "--quick"]) == 0
    db = Database(str(root / ".media-match.db"))
    assert db.count_files() == 5
    db.close()


def test_quick_scan_never_downgrades_analysed_row(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png")
    path = str(root / "a.png")
    db = Database(str(root / ".media-match.db"))
    db.upsert_file(path=path, size=1, mtime=1.0, sha1="deadbeef",
                   width=64, height=48, phash="0" * 16, keywords="",
                   description="already described")
    db.close()
    assert cli.main(["scan", str(root), "--quick"]) == 0
    db = Database(str(root / ".media-match.db"))
    row = db.get_file(path)
    db.close()
    assert row["sha1"] == "deadbeef" and row["phash"] == "0" * 16
    assert row["description"] == "already described"


def test_quick_scan_rejects_analysis_flags_and_window(tmp_path, capsys):
    root = _tree(tmp_path, n=1)
    for flag in ("--describe", "--refine", "--classify", "--reanalyze"):
        assert cli.main(["scan", str(root), "--quick", flag]) == 2, flag
    assert cli.main(["scan", str(root), "--quick", "--start-id", "1"]) == 2
    out = capsys.readouterr().err
    assert "--quick" in out


def test_quick_scan_missing_root(tmp_path, capsys):
    missing = tmp_path / "nope"
    assert cli.main(["scan", str(missing), "--quick"]) == 2
    captured = capsys.readouterr()
    assert "No folder" in captured.err
    assert captured.out == ""


# -- build_jobs: quick rows and the id window --------------------------------


def test_build_jobs_queues_quick_rows_even_when_unchanged(tmp_path):
    root = _tree(tmp_path, n=1)
    path = str(root / "0.png")
    stat = os.stat(path)
    db = Database(str(root / ".media-match.db"))
    db.add_quick_files([(path, stat.st_size, stat.st_mtime)])
    jobs, total, up_to_date = build_jobs(str(root), db, include_video=False,
                                         reanalyze=False)
    assert len(jobs) == 1 and total == 1 and up_to_date == 0
    # a quick row may already carry a caption; it must survive the scan
    db.set_description(jobs[0].path and db.get_file(path)["id"], "kept", "kw")
    jobs2, _t, _u = build_jobs(str(root), db, include_video=False,
                               reanalyze=False)
    db.close()
    assert jobs2[0].prev_description == "kept"
    assert jobs2[0].prev_keywords == "kw"


def test_build_jobs_window_selects_indexed_rows_only(tmp_path):
    root = _tree(tmp_path, n=3)
    db = Database(str(root / ".media-match.db"))
    assert cli.main(["scan", str(root), "--quick"]) == 0
    _write_image(root / "new.png")  # on disk, not in the index
    jobs, total, _up = build_jobs(str(root), db, include_video=False,
                                  reanalyze=False, start_id=2, stop_id=2)
    assert total == 1 and len(jobs) == 1
    assert jobs[0].path == str(root / "1.png")
    # window bounds are inclusive; new files are never part of a window
    jobs_all, total_all, _u = build_jobs(str(root), db, include_video=False,
                                         reanalyze=False, start_id=1)
    assert total_all == 3 and len(jobs_all) == 3
    db.close()


def test_build_jobs_window_with_limit_is_the_next_slice(tmp_path):
    root = _tree(tmp_path, n=5)
    db = Database(str(root / ".media-match.db"))
    assert cli.main(["scan", str(root), "--quick"]) == 0
    jobs, total, _u = build_jobs(str(root), db, include_video=False,
                                 reanalyze=False, start_id=3, limit=2)
    assert total == 2 and [j.path for j in jobs] == [
        str(root / "2.png"), str(root / "3.png")]
    db.close()


# -- duplicate grouping must ignore basic rows -------------------------------


def test_groupers_skip_quick_rows(tmp_path):
    root = _tree(tmp_path, n=2)
    db = Database(str(root / ".media-match.db"))
    db.add_quick_files([(str(root / "0.png"), 1, 1.0),
                        (str(root / "1.png"), 1, 1.0)])
    records = db.all_files()
    assert find_duplicate_groups(records) == []
    assert find_duplicate_groups(records, exact_phash=True) == []
    db.close()


def test_video_groupers_skip_quick_rows(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (root / name).write_bytes(b"x")
    db = Database(str(root / ".media-match.db"))
    db.add_quick_videos([(str(root / "a.mp4"), 1, 1.0),
                         (str(root / "b.mp4"), 1, 1.0)])
    assert find_video_duplicate_groups(db.all_videos_with_frames()) == []
    db.close()


def test_duplicate_groups_still_work_for_analysed_rows(tmp_path):
    root = _tree(tmp_path, n=2)
    db = Database(str(root / ".media-match.db"))
    for i in range(2):
        db.upsert_file(path=str(root / f"{i}.png"), size=10, mtime=1.0 + i,
                       sha1=f"h{i}", width=64, height=48, phash="ab" * 8,
                       description="", keywords="")
    groups = find_duplicate_groups(db.all_files())
    db.close()
    assert len(groups) == 1 and groups[0].size == 2


# -- id window across commands -----------------------------------------------


def _analysed_tree(tmp_path):
    root = _tree(tmp_path, n=3)
    db = Database(str(root / ".media-match.db"))
    for i in range(3):
        db.upsert_file(
            path=str(root / f"{i}.png"), size=100 + i, mtime=1.0 + i,
            sha1=f"sha{i}", width=64 + i, height=48, phash=f"{i}" * 16,
            description=f"photo {i}", keywords=f"kw{i}",
            captured_at=f"2021-01-0{i + 1}T10:00:00",
        )
    db.close()
    return root


def test_export_json_range(tmp_path, capsys):
    root = _analysed_tree(tmp_path)
    assert cli.main(["export", str(root), "--start-id", "2", "--stop-id", "3",
                     "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["path"] for r in data] == [str(root / "1.png"), str(root / "2.png")]


def test_export_alias_flags(tmp_path, capsys):
    root = _analysed_tree(tmp_path)
    assert cli.main(["export", str(root), "--startid", "2", "--stopid", "3"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["path"] for r in data] == [str(root / "1.png"), str(root / "2.png")]


def test_search_fts_respects_window(tmp_path, capsys):
    root = _analysed_tree(tmp_path)
    assert cli.main(["search", str(root), "photo", "--start-id", "3"]) == 0
    out = capsys.readouterr().out
    assert "2.png" in out and "0.png" not in out and "1.png" not in out


def test_export_metadata_window(tmp_path, capsys):
    root = _analysed_tree(tmp_path)
    assert cli.main(["export-metadata", str(root), "--start-id", "1",
                     "--stop-id", "1"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["path"] for r in data] == ["0.png"]


def test_clear_metadata_window_and_explicit_paths(tmp_path, capsys):
    root = _analysed_tree(tmp_path)
    # 1. window only, no items
    assert cli.main(["clear-metadata", str(root), "--start-id", "2",
                     "--stop-id", "3", "--yes"]) == 0
    db = Database(str(root / ".media-match.db"))
    assert db.get_file(str(root / "0.png"))["description"] == "photo 0"
    assert db.get_file(str(root / "1.png"))["description"] == ""
    db.close()
    # 2. explicit path outside the window is reported, not cleared
    assert cli.main(["clear-metadata", str(root), "0.png", "--start-id", "2",
                     "--yes"]) == 0
    assert "outside id window" in capsys.readouterr().out
    db = Database(str(root / ".media-match.db"))
    assert db.get_file(str(root / "0.png"))["description"] == "photo 0"
    db.close()


def test_import_metadata_outside_window(tmp_path, capsys):
    root = _analysed_tree(tmp_path)
    payload = json.dumps([
        {"path": "0.png", "description": "imported"},
        {"path": "1.png", "description": "imported"},
    ])
    source = tmp_path / "meta.json"
    source.write_text(payload)
    assert cli.main(["import-metadata", str(root), str(source),
                     "--start-id", "2"]) == 0
    out = capsys.readouterr().out
    assert "outside id window: 0.png" in out
    db = Database(str(root / ".media-match.db"))
    assert db.get_file(str(root / "0.png"))["description"] == "photo 0"
    assert db.get_file(str(root / "1.png"))["description"] == "imported"
    db.close()


def test_duplicates_window(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    db = Database(str(root / ".media-match.db"))
    # pair 1: ids 1-2 share a phash; pair 2: ids 3-4 share another
    for i, phash in enumerate(("ab" * 8, "ab" * 8, "cd" * 8, "cd" * 8)):
        _write_image(root / f"{i}.png")
        db.upsert_file(path=str(root / f"{i}.png"), size=10, mtime=1.0,
                       sha1=f"s{i}", width=64, height=48, phash=phash,
                       description="", keywords="")
    db.close()
    assert cli.main(["duplicates", str(root), "--start-id", "3", "--stop-id",
                     "4", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1
    paths = [data[0]["best"]] + [d["path"] for d in data[0]["duplicates"]]
    assert all(p.endswith(("2.png", "3.png")) for p in paths)


def test_refine_window_keeps_outside_tags(tmp_path):
    root = _analysed_tree(tmp_path)
    db = Database(str(root / ".media-match.db"))
    id1 = db.get_file(str(root / "0.png"))["id"]
    id2 = db.get_file(str(root / "1.png"))["id"]
    db.apply_refinement("image", [
        (id1, "alpha", "alpha", "photo 0", [("alpha", "object")]),
        (id2, "beta", "beta", "photo 1", [("beta", "object")]),
    ])
    from media_match.cli import _refine_all

    total = _refine_all(db, quiet=True, start_id=id1, stop_id=id1)
    assert total == 1
    tags1 = set(db.tags_for("image", id1))
    tags2 = set(db.tags_for("image", id2))
    db.close()
    # rebuilt inside the window: the tag survives but its facet is gone,
    # because raw_keywords ("alpha") no longer maps to a facet
    assert tags1 == {("alpha", None)}
    assert tags2 == {("beta", "object")}   # untouched outside the window


def test_describe_targets_window(tmp_path):
    root = _analysed_tree(tmp_path)
    db = Database(str(root / ".media-match.db"))
    db.conn.execute("UPDATE files SET description = ''")
    db.conn.commit()
    db.set_description(db.get_file(str(root / "0.png"))["id"], "has caption", "k")
    from media_match.analyze import describe_targets

    plan = describe_targets(db, str(root), include_videos=False,
                            start_id=2, stop_id=2)
    assert [j.media_id for j in plan.jobs] == [
        db.get_file(str(root / "1.png"))["id"]]
    assert plan.pending == {"image": 1}  # pending counts are window-scoped
    db.close()


def test_missing_embedding_window(tmp_path):
    root = _analysed_tree(tmp_path)
    db = Database(str(root / ".media-match.db"))
    assert [r["id"] for r in db.files_missing_embedding()] == [1, 2, 3]
    assert [r["id"] for r in db.files_missing_embedding(2, 3)] == [2, 3]
    db.close()


# -- validation ---------------------------------------------------------------


def test_inverted_window_exits_2(tmp_path, capsys):
    root = _analysed_tree(tmp_path)
    assert cli.main(["export", str(root), "--start-id", "5",
                     "--stop-id", "2"]) == 2
    assert "--start-id" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--start-id", "0"], ["--stop-id", "-1"],
                                  ["--start-id", "x"]])
def test_nonpositive_window_exits_2(tmp_path, args):
    root = _analysed_tree(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["export", str(root), *args])
    assert exc.value.code == 2
