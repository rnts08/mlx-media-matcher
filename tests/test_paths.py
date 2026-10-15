"""Paths/filenames with spaces and shell metacharacters must work everywhere.

Regression guard: the CLI must never hand a path to a shell (all subprocess calls
are list-based), so names like ``a & b.jpg``, ``quote'.jpg`` or ``café.jpg``
survive intact. If a path ever produced "file not found" here, it was being
split by the caller's shell before reaching us.
"""

import os

import numpy as np
import pytest
from PIL import Image

from media_match.cleanup import trash_files
from media_match.db import Database, FileRecord
from media_match.pipeline import build_jobs, run_scan, store_result


class _FakeEncoder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_images(self, arrays):
        return np.stack([np.ones(self.dim, dtype=np.float32) for _ in arrays])


def _write_image(path, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 30, 200)).save(path)


# names that break naive shell escaping / quoting on the command line
NASTY = [
    "plain.jpg",
    "with space.jpg",
    "amp&and.jpg",
    "quote'.jpg",
    'dq".jpg',
    "hash#.jpg",
    "semi;.jpg",
    "dollar$.jpg",
    "comma,.jpg",
    "café.jpg",
    "emoji\U0001F600.jpg",
]


def test_special_char_paths_round_trip(tmp_path):
    root = tmp_path / "Timh & Dung"
    for name in NASTY:
        _write_image(root / name)
    db = Database(str(tmp_path / "t.db"))

    jobs, total, _ = build_jobs(str(root), db, include_video=False, reanalyze=False)
    assert total == len(NASTY) and len(jobs) == len(NASTY)

    encoder = _FakeEncoder()
    collected = []
    run_scan(jobs, encoder, interval=1.0, max_frames=120,
             batch_size=4, workers=2, on_result=collected.append)
    for r in collected:
        store_result(db, r)

    stored = {r.path for r in db.all_files()}
    expected = {str(root / name) for name in NASTY}
    assert stored == expected
    assert db.count_files() == len(NASTY)


def test_special_char_trash(tmp_path):
    pytest.importorskip("send2trash")
    root = tmp_path / "Timh & Dung"
    target = root / "delete me & stuff.jpg"
    _write_image(target)
    rec = FileRecord(
        id=1, path=str(target), size=10, mtime=1.0, sha1="x", width=64,
        height=48, phash="0" * 16, description="", keywords="",
    )
    trashed, failed = trash_files([rec], out=lambda _m: None)
    assert not failed and len(trashed) == 1
    assert not os.path.exists(target)
