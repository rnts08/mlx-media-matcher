import numpy as np

from media_match.db import VideoRecord
from media_match.video_groups import (
    _candidate_pairs,
    find_video_duplicate_groups,
    find_video_similarity_groups,
    pair_ratio,
)


def test_pair_ratio_identical():
    hashes = ["0000000000000001", "0000000000000002",
              "0000000000000004", "0000000000000008"]
    ratio, dist = pair_ratio(hashes, hashes)
    assert ratio == 1.0
    assert dist == 0


def test_pair_ratio_is_not_all_pairs():
    # b is a 2x-upsampled sampling of a: matching frames sit at the positions
    # linspace(0, 7, 4) picks, so aligned matching finds them but an
    # every-frame-to-every-frame matcher would not need to.
    a = ["0000000000000001", "0000000000000002",
         "0000000000000004", "0000000000000008"]
    b = [a[0], "ffffffffffffffff", a[1], "fffffffffffffffe",
         "fffffffffffffffd", a[2], "fffffffffffffffc", a[3]]
    ratio, _ = pair_ratio(a, b)
    assert ratio == 1.0


def test_pair_ratio_unrelated():
    a = ["0000000000000001", "0000000000000002"]
    b = ["ffffffffffffff00", "ffffffffffff00ff"]
    ratio, dist = pair_ratio(a, b)
    assert ratio == 0.0
    assert dist > 12


def _v(id, path, w, h, bitrate, hashes, *, duration=1.0, sha1=None):
    return VideoRecord(
        id=id, path=path, size=w * h, mtime=0.0, sha1=sha1 or path,
        duration=duration, width=w, height=h, fps=10.0, codec="h264",
        bitrate=bitrate, frame_count=len(hashes), signature=min(hashes) if hashes else "",
        description="", keywords="", phashes=list(hashes),
    )


def test_find_video_duplicate_groups_prefers_resolution():
    hashes = ["0000000000000001", "0000000000000002",
              "0000000000000004", "0000000000000008"]
    hd = _v(1, "/movies/a_1080.mp4", 1920, 1080, 5_000_000, hashes)
    sd = _v(2, "/movies/a_480.mp4", 854, 480, 800_000, hashes)
    other = _v(3, "/movies/other.mp4", 1920, 1080, 5_000_000,
               ["fffffffffffffff1", "fffffffffffffff2",
                "fffffffffffffff4", "fffffffffffffff8"])
    groups = find_video_duplicate_groups([hd, sd, other], frame_threshold=12,
                                         merge_ratio=0.6)
    assert len(groups) == 1
    assert groups[0].best.path == "/movies/a_1080.mp4"
    assert [r.path for r, _ in groups[0].others] == ["/movies/a_480.mp4"]


def test_candidate_pairs_shared_band():
    a = ["0000000000000001"]
    b = ["0000000000000003"]   # differs in 1 bit -> shares every band
    c = ["ffffffffffffffff"]
    pairs = _candidate_pairs([a, b, c])
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (1, 2) not in pairs


def test_duplicate_group_forced_lsh_path():
    # full_compare_limit=0 forces the band-bucket candidate path; an exact
    # re-encode must still be found, and an unrelated video never grouped.
    hashes = ["0000000000000001", "0000000000000002",
              "0000000000000004", "0000000000000008"]
    hd = _v(1, "/movies/a_1080.mp4", 1920, 1080, 5_000_000, hashes)
    sd = _v(2, "/movies/a_480.mp4", 854, 480, 800_000, hashes)
    other = _v(3, "/movies/other.mp4", 1920, 1080, 5_000_000,
               ["fffffffffffffff1", "fffffffffffffff2",
                "fffffffffffffff4", "fffffffffffffff8"])
    groups = find_video_duplicate_groups([hd, sd, other], frame_threshold=12,
                                         merge_ratio=0.6, full_compare_limit=0)
    assert len(groups) == 1
    assert groups[0].best.path == "/movies/a_1080.mp4"
    assert [r.path for r, _ in groups[0].others] == ["/movies/a_480.mp4"]


def test_find_video_similarity_groups():
    a = _v(1, "/v/a.mp4", 1920, 1080, 1, ["00"])
    b = _v(2, "/v/b.mp4", 1920, 1080, 1, ["01"])
    c = _v(3, "/v/c.mp4", 1920, 1080, 1, ["02"])
    vec_a = np.array([[1.0, 0.0]], dtype=np.float32)
    vec_b = np.array([[0.98, 0.02]], dtype=np.float32)
    vec_b /= np.linalg.norm(vec_b)
    vec_c = np.array([[0.0, 1.0]], dtype=np.float32)
    embeddings = {1: vec_a, 2: vec_b, 3: vec_c}
    groups = find_video_similarity_groups([a, b, c], embeddings, threshold=0.85)
    assert len(groups) == 1
    assert {groups[0].best.id, groups[0].others[0][0].id} == {1, 2}
