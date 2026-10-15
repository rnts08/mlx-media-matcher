import numpy as np

from media_match.video_search import best_moment, format_timestamp, score_video


def test_best_moment_picks_best_frame_and_timestamp():
    query = np.array([1.0, 0.0], dtype=np.float32)
    frames = {
        0.0: np.array([0.0, 1.0], dtype=np.float32),   # orthogonal
        1.0: np.array([0.9, 0.1], dtype=np.float32),   # matches
    }
    score, ts = best_moment(query, frames)
    assert ts == 1.0
    assert abs(score - float(query @ frames[1.0])) < 1e-6


def test_best_moment_empty():
    assert best_moment(np.ones(2, np.float32), {}) == (0.0, 0.0)


def test_score_video():
    query = np.array([1.0, 0.0], dtype=np.float32)
    frames = {0.0: np.array([0.5, 0.5], dtype=np.float32)}
    assert score_video(query, frames) == float(query @ frames[0.0])


def test_format_timestamp():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(65) == "1:05"
    assert format_timestamp(3725) == "1:02:05"
