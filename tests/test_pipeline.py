import numpy as np
import pytest
from PIL import Image

from media_match.db import Database
from media_match.pipeline import build_jobs, prepare, run_scan, store_result
from tests.conftest import requires_ffmpeg


class FakeEncoder:
    """Stands in for ClipEncoder so pipeline tests need no MLX/GPU."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls = 0

    def embed_images(self, arrays):
        self.calls += 1
        return np.stack([np.ones(self.dim, dtype=np.float32) for _ in arrays])


def _write_image(path, size=(64, 48), color=(120, 30, 200)):
    Image.new("RGB", size, color).save(path)


def test_build_jobs_finds_and_skips_images(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png")
    db = Database(str(tmp_path / "t.db"))

    jobs, total, up_to_date = build_jobs(str(root), db, include_video=False,
                                          reanalyze=False)
    assert total == 1 and len(jobs) == 1 and up_to_date == 0
    assert jobs[0].kind == "image"

    result = prepare(jobs[0], interval=1.0, max_frames=120)
    result.vectors = [np.ones(4, dtype=np.float32)]
    store_result(db, result)

    jobs2, total2, up_to_date2 = build_jobs(str(root), db, include_video=False,
                                             reanalyze=False)
    assert jobs2 == [] and up_to_date2 == 1


def test_run_scan_embeds_images_in_batches(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    for i in range(5):
        _write_image(root / f"{i}.png", color=(i * 20, 10, 10))
    db = Database(str(tmp_path / "t.db"))
    jobs, *_ = build_jobs(str(root), db, include_video=False, reanalyze=False)

    encoder = FakeEncoder()
    results = run_scan(jobs, encoder, interval=1.0, max_frames=120,
                       batch_size=2, workers=3)
    assert len(results) == 5
    assert all(len(r.vectors) == 1 for r in results)
    assert encoder.calls == 3  # ceil(5/2)
    for r in results:
        store_result(db, r)
    assert db.count_files() == 5


@requires_ffmpeg
def test_run_scan_mixes_images_and_videos(tmp_path, make_video):
    root = tmp_path / "tree"
    root.mkdir()
    _write_image(root / "a.png")
    video = make_video("clip.mp4", seconds=4.0)
    (root / "clip.mp4").write_bytes(open(video, "rb").read())

    db = Database(str(tmp_path / "t.db"))
    jobs, total, _ = build_jobs(str(root), db, include_video=True, reanalyze=False)
    assert total == 2
    kinds = {j.kind for j in jobs}
    assert kinds == {"image", "video"}

    encoder = FakeEncoder()
    results = run_scan(jobs, encoder, interval=1.0, max_frames=120,
                       batch_size=4, workers=4)
    assert len(results) == 2
    for r in results:
        store_result(db, r)

    assert db.count_files() == 1
    assert db.count_videos() == 1
    video_rec = db.all_videos()[0]
    assert video_rec.frame_count == 4          # ~1 fps over 4 seconds
    assert len(db.video_frames(video_rec.id)) == 4
    # every keyframe got an embedding from the shared batch queue
    assert len(db.video_embeddings()[video_rec.id]) == 4
