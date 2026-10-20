"""The describe stage: one unified, pipelined queue for images and videos.

These tests run without MLX or a VLM — a fake describer stands in for the GPU
model — so they verify the plumbing: target selection, duplicate shadowing,
CPU prefetch (including ffmpeg stills), storage and error reporting.
"""

import pytest
from PIL import Image

from media_match.analyze import describe_targets, load_frame, run_describe_stage
from media_match.db import Database
from media_match.pipeline import build_jobs, prepare, run_scan, store_result
from tests.conftest import requires_ffmpeg

import numpy as np


class _FakeEncoder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_images(self, arrays):
        return np.stack([np.ones(self.dim, dtype=np.float32) for _ in arrays])


class FakeDescriber:
    """Stands in for the VLM; records what it was fed."""

    def __init__(self, fail_on: set[str] | None = None):
        self.seen: list[str] = []
        self.fail_on = fail_on or set()

    def describe(self, frame) -> tuple[str, list[str]]:
        label = f"{frame.width}x{frame.height}"
        self.seen.append(label)
        if label in self.fail_on:
            raise RuntimeError("boom")
        return f"a {label} photo", ["test", "frame"]


def _write_image(path, size=(64, 48), color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _index_tree(root, db):
    jobs, *_ = build_jobs(str(root), db, include_video=True, reanalyze=False)
    collected = []
    run_scan(jobs, _FakeEncoder(), interval=1.0, max_frames=120,
             batch_size=4, workers=2, on_result=collected.append)
    for r in collected:
        store_result(db, r)


def test_targets_unify_images_and_videos(tmp_path, make_video):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png")
    (root / "clip.mp4").write_bytes(open(make_video("c.mp4", seconds=2.0), "rb").read())

    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)

    plan = describe_targets(db, str(root), include_videos=True)
    kinds = {j.kind for j in plan.jobs}
    assert kinds == {"image", "video"}
    assert plan.missing == 0 and plan.shadowed == 0
    assert plan.pending == {"image": 1, "video": 1}


def test_stage_captions_images_and_videos(tmp_path, make_video):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png")
    (root / "clip.mp4").write_bytes(open(make_video("c.mp4", seconds=2.0), "rb").read())

    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    plan = describe_targets(db, str(root), include_videos=True)

    describer = FakeDescriber()
    done = []
    n = run_describe_stage(plan.jobs, describer, db, workers=2,
                           on_result=lambda job, d, k: done.append(job))
    assert n == 2 and len(done) == 2

    img = db.all_files()[0]
    assert img.description.startswith("a ")
    assert img.keywords == "test, frame"
    vid = db.all_videos()[0]
    assert vid.description.startswith("a ")
    # a second run has nothing left to do
    plan2 = describe_targets(db, str(root), include_videos=True)
    assert plan2.jobs == []


def test_stage_reports_and_survives_errors(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png", color=(120, 30, 200))
    _write_image(root / "b.png", color=(200, 30, 120))
    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    plan = describe_targets(db, str(root), include_videos=False)

    describer = FakeDescriber(fail_on={"64x48"})  # every frame fails
    errors = []
    n = run_describe_stage(plan.jobs, describer, db, workers=1,
                           on_error=lambda p, e: errors.append(p))
    assert n == 0 and len(errors) == 2
    # nothing was written
    assert all(not r.description for r in db.all_files())


def test_load_frame_decodes_image_and_video_still(tmp_path, make_video):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png", size=(80, 60))
    (root / "clip.mp4").write_bytes(open(make_video("c.mp4", seconds=2.0), "rb").read())

    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    plan = describe_targets(db, str(root), include_videos=True)
    by_kind = {j.kind: j for j in plan.jobs}

    frame = load_frame(by_kind["image"])
    assert frame.size == (80, 60) and frame.mode == "RGB"

    # ffmpeg scales stills to the 448x448 VLM frame size
    still = load_frame(by_kind["video"])
    assert still.size == (448, 448) and still.mode == "RGB"


def test_missing_media_are_skipped(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "stay.png", color=(120, 30, 200))
    _write_image(root / "gone.png", color=(200, 30, 120))
    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)

    import os

    os.unlink(root / "gone.png")
    plan = describe_targets(db, str(root), include_videos=False)
    assert plan.missing == 1
    assert [j.label for j in plan.jobs] == ["stay.png"]


def test_duplicate_copies_are_shadowed(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "big.png", size=(200, 150))
    _write_image(root / "small.png", size=(50, 40))  # same color -> same pHash
    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    assert db.count_files() == 2

    plan = describe_targets(db, str(root), include_videos=False)
    assert plan.shadowed == 1
    assert len(plan.jobs) == 1
    # the best-resolution copy is the one queued
    assert plan.jobs[0].label == "big.png"


def test_describe_only_queues_undescribed(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png")
    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    plan = describe_targets(db, str(root), include_videos=False)
    run_describe_stage(plan.jobs, FakeDescriber(), db, workers=1)

    plan2 = describe_targets(db, str(root), include_videos=False)
    assert plan2.jobs == [] and plan2.shadowed == 0


# -- live telemetry ---------------------------------------------------------


def _bar(tmp_path, total):
    """A bar that is never started: no thread, no drawing, just the state."""
    import io

    from media_match.progress import ProgressBar

    return ProgressBar("describing", str(tmp_path), total,
                       stream=io.StringIO(), enabled=True)


def _write_noise(path, seed):
    """Distinct images with distinct pHashes; solid colours collide."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)


def test_stage_reports_progress_telemetry(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    for i in range(5):
        _write_noise(root / f"{i}.png", seed=i)
    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    plan = describe_targets(db, str(root), include_videos=False)
    assert len(plan.jobs) == 5

    bar = _bar(tmp_path, len(plan.jobs))
    n = run_describe_stage(plan.jobs, FakeDescriber(), db, workers=2,
                           window=4, telemetry=bar)

    assert n == 5 == bar.state.done
    assert bar.state.errors == 0
    assert bar.state.decoding == [], "finished names must leave the display"
    # The prefetch window is reported in items, not bytes.
    assert bar.state.ring_label == "window"
    assert bar.state.ring_cap == 4
    assert bar.state.ring_bytes is False
    # GPU time is measured around the describe calls, CPU around the stage.
    assert bar.state.gpu_seconds > 0
    assert bar.state.cpu_seconds > 0


def test_stage_reports_errors_without_counting_them_as_described(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png", color=(120, 30, 200))
    _write_image(root / "b.png", color=(200, 30, 120))
    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    plan = describe_targets(db, str(root), include_videos=False)

    bar = _bar(tmp_path, len(plan.jobs))
    n = run_describe_stage(plan.jobs, FakeDescriber(fail_on={"64x48"}), db,
                           workers=1, telemetry=bar)

    assert n == 0
    assert bar.state.errors == 2
    assert bar.state.done == 0, "a failed caption is not a completed item"


def test_window_is_reported_after_the_stage(tmp_path):
    """The fill/drain bookkeeping must not leave a stale window behind."""
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png")
    db = Database(str(tmp_path / "t.db"))
    _index_tree(root, db)
    plan = describe_targets(db, str(root), include_videos=False)

    bar = _bar(tmp_path, len(plan.jobs))
    run_describe_stage(plan.jobs, FakeDescriber(), db, workers=1, window=4,
                       telemetry=bar)
    assert bar.state.ring_used == 0
    # The window is clamped to the job count but never below two.
    assert bar.state.ring_cap == 2
