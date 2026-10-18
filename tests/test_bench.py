"""The benchmark: it has to be honest, because people set flags from it.

A benchmark that recommends the nominal winner regardless of the numbers is
worse than no benchmark — it turns a coin flip into a tuning decision.
"""

import os

import pytest

from media_match.bench import (
    CANDIDATES,
    ConfigResult,
    StageResult,
    cheapest,
    human_seconds,
    indistinguishable,
    make_corpus,
    measure_cpu,
    measure_disk,
    recommend,
)


# -- corpus ---------------------------------------------------------------------


def test_make_corpus_writes_the_requested_number_of_readable_images(tmp_path):
    from PIL import Image

    paths = make_corpus(str(tmp_path), 6)
    assert len(paths) == 6
    assert len({p for p in paths}) == 6
    for path in paths:
        assert os.path.getsize(path) > 0
        with Image.open(path) as img:
            img.verify()  # a corrupt corpus would make every number a lie


def test_make_corpus_is_deterministic(tmp_path):
    # The same corpus every run is what makes numbers comparable between runs
    # and between machines.
    a = make_corpus(str(tmp_path / "a"), 3)
    b = make_corpus(str(tmp_path / "b"), 3)
    assert [os.path.basename(p) for p in a] == [os.path.basename(p) for p in b]
    assert [open(p, "rb").read() for p in a] == [open(p, "rb").read() for p in b]


def test_make_corpus_spans_realistic_shapes(tmp_path):
    from PIL import Image

    paths = make_corpus(str(tmp_path), 4)
    sizes = []
    for path in paths:
        with Image.open(path) as img:
            sizes.append(img.size)
    # a mix of a phone photo and something small, not 4 copies of one size
    assert max(w * h for w, h in sizes) > 4_000_000
    assert min(w * h for w, h in sizes) < 1_000_000


# -- stages ---------------------------------------------------------------------


def test_measure_disk_reads_every_file(tmp_path):
    paths = make_corpus(str(tmp_path), 4)
    got = measure_disk(paths, readers=2)
    assert isinstance(got, StageResult)
    assert got.seconds > 0
    assert "MB/s" in got.detail
    assert "4 file(s)" in got.detail


def test_measure_cpu_handles_bad_files_without_raising(tmp_path):
    from PIL import Image

    good = tmp_path / "good.jpg"
    Image.new("RGB", (64, 48), (1, 2, 3)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    # A corrupt file in the tree must not abort the benchmark — real trees
    # have them, and they are exactly what 'check' is for.
    got = measure_cpu([str(good)], workers=2)
    assert got.seconds > 0
    assert "img/s" in got.detail


# -- recommendation -------------------------------------------------------------


def _cfg(batch, io, cpu, wall, items=10):
    return ConfigResult(batch=batch, io_workers=io, workers=cpu, items=items,
                        wall=wall, gpu=wall * 0.5, cpu=wall * 0.4)


def test_recommend_picks_the_fastest_working_configuration():
    configs = [_cfg(8, 1, 2, 2.0), _cfg(16, 2, 4, 1.0), _cfg(32, 4, 8, 1.5)]
    assert recommend(configs).batch == 16


def test_recommend_ignores_configurations_that_failed():
    failed = _cfg(64, 8, 16, 0.0, items=0)
    failed.note = "boom"
    configs = [_cfg(16, 2, 4, 1.0), failed]
    assert recommend(configs).batch == 16


def test_recommend_returns_none_when_nothing_worked():
    assert recommend([_cfg(8, 1, 2, 0.0, items=0)]) is None


def test_indistinguishable_groups_configurations_within_the_noise_floor():
    # Measured run-to-run spread on a small corpus reaches ~10%; 1.0s and
    # 1.05s are the same result as far as this benchmark can tell.
    best = _cfg(16, 2, 4, 1.00)
    others = [_cfg(8, 1, 2, 2.00), _cfg(16, 4, 8, 1.05), _cfg(32, 4, 8, 1.02)]
    tied = indistinguishable(best, others + [best])
    assert {c.batch for c in tied} == {16, 32}
    assert _cfg(8, 1, 2, 2.00) not in tied


def test_indistinguishable_excludes_failed_configurations():
    best = _cfg(16, 2, 4, 1.0)
    broken = _cfg(32, 4, 8, 1.0, items=0)
    broken.note = "boom"
    assert indistinguishable(best, [best, broken]) == [best]


def test_cheapest_prefers_fewest_threads_then_smallest_batch():
    tied = [_cfg(32, 4, 8, 1.0), _cfg(8, 2, 4, 1.05), _cfg(16, 2, 4, 1.02)]
    pick = cheapest(tied)
    assert (pick.workers, pick.io_workers, pick.batch) == (4, 2, 8)


# -- formatting -----------------------------------------------------------------


@pytest.mark.parametrize("seconds,expected", [
    (0.4, "0.40s"),
    (12.0, "12s"),
    (204.0, "3.4 min"),
    (7500.0, "2h05m"),
    (180000.0, "2 days 2h"),
])
def test_human_seconds_never_shows_a_bare_zero(seconds, expected):
    # An estimate of "0.0 h" is worse than no estimate.
    assert human_seconds(seconds) == expected


# -- CLI ------------------------------------------------------------------------


def test_about_bench_runs_without_a_folder(capsys, monkeypatch, tmp_path):
    from media_match import cli

    class FakeReport:
        corpus_items = 8
        corpus_bytes = 1234
        stages = [StageResult("disk read", 0.01, "1 thread(s): 100 MB/s")]
        configs = [_cfg(8, 2, 4, 1.0)]
        best = _cfg(16, 2, 4, 1.0)
        ties = [_cfg(16, 2, 4, 1.0)]
        recommended = _cfg(16, 2, 4, 1.0)
        note = ""

    monkeypatch.setattr("media_match.bench.run_benchmark", lambda **kw: FakeReport())
    assert cli.main(["about", "bench"]) == 0
    out = capsys.readouterr().out
    assert "--batch 16" in out
    assert "--io-workers 2" in out
    assert "--workers 4" in out
    assert "Nothing was indexed" in out


def test_about_without_bench_still_prints_the_blurb(capsys):
    from media_match import cli

    assert cli.main(["about"]) == 0
    out = capsys.readouterr().out
    assert "Local photo-library analyzer" in out
    assert "--batch" not in out


def test_about_bench_reports_an_empty_tree_instead_of_crashing(capsys, tmp_path):
    from media_match import cli

    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["about", "bench", str(empty)]) == 1
    assert "no media found" in capsys.readouterr().out


def test_every_candidate_is_a_valid_configuration():
    for batch, io_workers, workers in CANDIDATES:
        assert batch >= 1 and io_workers >= 1 and workers >= 1