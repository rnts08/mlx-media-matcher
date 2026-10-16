import numpy as np
from PIL import Image

from media_match import cli
from media_match.db import Database
from media_match.pipeline import ScanStats, build_jobs, run_scan


class FakeEncoder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_images(self, arrays):
        return np.stack([np.ones(self.dim, dtype=np.float32) for _ in arrays])


def _img(path, color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)


def test_scan_stats_measured(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    for i in range(3):
        _img(root / f"{i}.png", (i * 30, 0, 0))
    db = Database(str(tmp_path / "t.db"))
    jobs, *_ = build_jobs(str(root), db, include_video=False, reanalyze=False)

    stats = ScanStats()
    run_scan(jobs, FakeEncoder(), interval=1.0, max_frames=120,
             batch_size=2, workers=1, stats=stats)

    assert stats.items == 3
    assert stats.frames == 3
    assert stats.bytes_read > 0
    assert stats.wall > 0 and stats.cpu > 0
    assert stats.gpu >= 0
    db.close()


def test_record_and_summarize_accumulates(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_scan(items=10, frames=12, bytes_read=1000, wall=2.0, cpu=1.5, gpu=1.8)
    db.record_scan(items=5, frames=6, bytes_read=500, wall=1.0, cpu=0.7, gpu=0.9)

    s = db.scan_summary()
    assert s["scans"] == 2
    assert s["items"] == 15 and s["frames"] == 18 and s["bytes"] == 1500
    assert abs(s["wall"] - 3.0) < 1e-6
    assert abs(s["cpu"] - 2.2) < 1e-6
    assert abs(s["gpu"] - 2.7) < 1e-6
    assert s["last_scan_at"]
    db.close()


def test_extension_counts(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    for name in ("a.jpg", "b.png", "c.jpg"):
        _img(root / name)
    db = Database(str(tmp_path / "t.db"))
    for name, size in (("a.jpg", 100), ("b.png", 200), ("c.jpg", 300)):
        db.upsert_file(
            path=str(root / name), size=size, mtime=1.0, sha1=name, width=1,
            height=1, phash="0" * 16, description="", keywords="",
        )
    counts, sizes = db.image_extensions()
    assert dict(counts) == {".jpg": 2, ".png": 1}
    assert sizes[".jpg"] == 400 and sizes[".png"] == 200
    db.close()


def test_stats_cli_reports_breakdown(tmp_path, capsys):
    root = tmp_path / "tree"
    root.mkdir()
    _img(root / "a.jpg")
    _img(root / "b.png")
    db = Database(str(tmp_path / ".media-match.db"))
    db.record_scan(items=2, frames=2, bytes_read=1000, wall=1.0, cpu=0.5, gpu=0.6)
    db.close()

    assert cli.main(["stats", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "compute:" in out and "I/O" in out
    assert "cpu" in out and "gpu" in out
    assert "formats:" in out
    assert "embeddings" in out
