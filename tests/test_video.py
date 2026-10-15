import numpy as np
import pytest

from tests.conftest import requires_ffmpeg
from media_match.video import (
    MAX_FRAMES,
    extract_frames,
    extract_still_png,
    probe,
    sample_count,
)


def test_sample_count_one_per_second():
    assert sample_count(10.0) == 10


def test_sample_count_floor_and_cap():
    assert sample_count(0.5) >= 3              # very short clip still gets a few
    assert sample_count(0.5, min_frames=3) == 3
    assert sample_count(10_000.0) == MAX_FRAMES  # long clip samples sparser


def test_sample_count_zero_duration():
    assert sample_count(0.0) == 0


@requires_ffmpeg
def test_probe_reads_metadata(make_video):
    meta = probe(make_video("probe.mp4", seconds=2.0, size="320x240", fps=10))
    assert (meta.width, meta.height) == (320, 240)
    assert abs(meta.duration - 2.0) < 0.2
    assert abs(meta.fps - 10) < 0.5
    assert meta.codec == "h264"
    assert meta.bitrate > 0


@requires_ffmpeg
def test_extract_frames_count_and_shape(make_video):
    path = make_video("frames.mp4", seconds=4.0, size="320x240", fps=10)
    meta = probe(path)
    frames = extract_frames(path, meta, count=4)
    assert len(frames) == 4
    ts, arr = frames[0]
    assert ts == 0.0
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8
    assert frames[-1][0] > 2.0  # last sample is near the end


@requires_ffmpeg
def test_extract_still_png(make_video):
    path = make_video("still.mp4", seconds=2.0)
    png = extract_still_png(path, at=1.0, size=128)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
