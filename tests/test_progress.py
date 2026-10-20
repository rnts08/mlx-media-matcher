"""Live progress: correct numbers, and silent when there is no terminal."""

import io
import threading
import time

import pytest

from media_match.progress import (
    PipelineState,
    ProgressBar,
    bar,
    human_bytes,
    human_time,
    render,
    short_name,
)


class _FakeTTY(io.StringIO):
    """A StringIO that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


# -- formatting -----------------------------------------------------------------


@pytest.mark.parametrize("n,expected", [
    (0, "0B"), (900, "900B"), (1024, "1KB"), (1536, "2KB"),
    (1024 ** 2, "1MB"), (int(1.5 * 1024 ** 3), "1.5GB"),
])
def test_human_bytes(n, expected):
    assert human_bytes(n) == expected


@pytest.mark.parametrize("seconds,expected", [
    (0.4, "0.4s"), (12.0, "12.0s"), (95.0, "1m35s"), (7500.0, "2h05m"),
])
def test_human_time_never_shows_a_bare_zero(seconds, expected):
    assert human_time(seconds) == expected


def test_human_time_clamps_negatives():
    # A clock that reads slightly backwards must not render "-0.1s".
    assert human_time(-1.0) == "0.0s"


@pytest.mark.parametrize("fraction,expected", [
    (0.0, "░░░░░░░░░░"), (1.0, "██████████"), (0.5, "█████░░░░░"),
])
def test_bar_is_exactly_the_requested_width(fraction, expected):
    assert bar(fraction, 10) == expected


@pytest.mark.parametrize("bad", [-1.0, -99.0, 1.5, 2.0])
def test_bar_clamps_out_of_range_fractions(bad):
    # An eta computed past the end must not draw more blocks than the width.
    assert len(bar(bad, 10)) == 10


def test_short_name_keeps_the_filename_when_truncating():
    deep = "/root/" + "/".join(f"folder{i}" for i in range(20)) + "/IMG_4471.jpg"
    shown = short_name(deep, "/root", limit=24)
    assert len(shown) <= 24
    assert shown.endswith("IMG_4471.jpg")  # the part you care about survives


def test_short_name_leaves_short_paths_alone():
    assert short_name("/root/a.jpg", "/root") == "a.jpg"


# -- rendering -----------------------------------------------------------------


def _state(**kw) -> PipelineState:
    state = PipelineState(total=100, started=time.perf_counter())
    for key, value in kw.items():
        setattr(state, key, value)
    return state


def test_render_shows_progress_rate_and_eta():
    state = _state(done=50, total=100)
    state.started = time.perf_counter() - 10  # 50 items in ~10s
    text = render(state, "/root", "indexing", width=100)
    head = text.split("\n")[0]
    assert "50/100" in head
    assert "50%" in head
    assert "/s" in head
    assert "eta" in head


def test_slow_rates_keep_a_decimal():
    """Describing runs at ~1/s; rounding that to "0/s" looks like a stall."""
    state = _state(done=2, total=10)
    state.started = time.perf_counter() - 10  # 0.2 items/s
    head = render(state, "/root", "describing", width=100).split("\n")[0]
    assert "0.2/s" in head, head
    assert "0/s" not in head


def test_render_does_not_show_an_absurd_rate_on_the_first_frame():
    # elapsed is microseconds on frame one; without a floor this printed
    # "648,493/s" and a GPU utilisation of 739,281,703%.
    state = _state(done=1, total=100)
    state.started = time.perf_counter()
    text = render(state, "/root", "indexing", width=100)
    assert "/s" not in text.split("\n")[0]
    assert "eta" not in text.split("\n")[0]
    assert "1000000%" not in text and "739281703" not in text


def test_render_shows_the_names_of_files_in_flight():
    state = _state(done=1, decoding=["/root/Photos/IMG_4471.jpg"])
    text = render(state, "/root", "indexing", width=100)
    assert "Photos/IMG_4471.jpg" in text


def test_render_falls_back_to_reading_names_then_idle():
    state = _state(done=1, reading=["/root/a.jpg"])
    assert "a.jpg" in render(state, "/root", "indexing", width=100)
    state2 = _state(done=1)
    assert "idle" in render(state2, "/root", "indexing", width=100)


def test_render_shows_buffer_cpu_and_gpu():
    state = _state(done=1, ring_used=512 * 1024 * 1024, ring_cap=1024 ** 3,
                   cpu_seconds=38.2, gpu_seconds=11.4, bytes_read=1200 * 1024 ** 2)
    text = render(state, "/root", "indexing", width=120)
    assert "ring 512MB/1.0GB" in text
    assert "50%" in text
    assert "cpu 38.2s" in text
    assert "gpu 11.4s" in text
    assert "read 1.2GB" in text


def test_ring_capacity_keeps_its_decimal():
    # Rounding a 1GB buffer to "1GB" is fine, but 1.5GB must not read "2GB":
    # that overstates a full buffer by a third.
    assert human_bytes(1.5 * 1024 ** 3) == "1.5GB"
    assert human_bytes(1024 ** 3) == "1.0GB"


def test_render_omits_the_ring_until_it_is_known():
    # Before the first sample there is no capacity to compare against; showing
    # "ring 0B/0B 0%" would be noise.
    text = render(_state(done=1), "/root", "indexing", width=100)
    assert "ring" not in text


def test_render_shows_a_window_measured_in_items():
    """Describe's prefetch window is a count of frames, not bytes."""
    state = _state(done=2, ring_used=3, ring_cap=8)
    state.ring_label = "window"
    state.ring_bytes = False
    text = render(state, "/root", "describing", width=120)
    assert "window 3/8" in text and "38%" in text
    assert "B/" not in text, "a frame count must not be rendered as bytes"


def test_set_window_switches_the_buffer_to_items():
    bar = ProgressBar("describing", "/root", total=4, stream=io.StringIO(),
                      enabled=True)
    bar.set_ring(1024, 2048)
    bar.set_window(5, 8)
    assert bar.state.ring_label == "window"
    assert bar.state.ring_bytes is False
    assert (bar.state.ring_used, bar.state.ring_cap) == (5, 8)


def test_render_reports_gpu_percentage_and_errors():
    state = _state(done=1, gpu_seconds=5.0)
    state.started = time.perf_counter() - 10
    text = render(state, "/root", "indexing", width=120)
    assert "50%)" in text
    state.errors = 3
    assert "errors 3" in render(state, "/root", "indexing", width=120)


def test_render_never_exceeds_the_terminal_width():
    state = _state(done=1, decoding=["/root/" + "x" * 300 + "/a.jpg"])
    for line in render(state, "/root", "indexing", width=60).split("\n"):
        assert len(line) <= 60


def test_render_with_no_total_does_not_divide_by_zero():
    state = PipelineState(total=0, started=time.perf_counter())
    text = render(state, "/root", "indexing", width=100)
    assert "0/0" in text
    assert "eta" not in text.split("\n")[0]


# -- ProgressBar ----------------------------------------------------------------


def test_disabled_bar_writes_nothing():
    stream = io.StringIO()  # isatty() is False
    bar = ProgressBar("indexing", "/root", 10, stream=stream)
    assert bar.enabled is False
    bar.start()
    bar.set_total(10)
    bar.note_completed()
    bar.note_reading("/root/a.jpg")
    bar.set_ring(1, 2)
    bar.note_gpu(0.5)
    bar.draw()
    bar.stop()
    assert stream.getvalue() == ""


def test_enabled_bar_draws_in_place_and_ends_with_a_newline():
    stream = _FakeTTY()
    bar = ProgressBar("indexing", "/root", 10, stream=stream).start()
    bar.note_decoding("/root/a.jpg")
    bar.note_completed()
    bar.set_ring(5, 10)
    bar.note_gpu(0.25)
    bar.draw()
    bar.stop()
    out = stream.getvalue()
    assert "indexing" in out
    assert "a.jpg" in out
    assert out.endswith("\n")          # the cursor is left on a clean line
    assert "\r" in out                 # rewritten in place, not appended


def test_bar_counts_completions_and_errors():
    bar = ProgressBar("indexing", "/root", 3, stream=io.StringIO())
    bar.note_completed()
    bar.note_completed()
    bar.note_error()
    assert bar.state.done == 2
    assert bar.state.errors == 1


def test_bar_accumulates_gpu_time_and_replaces_cpu_time():
    bar = ProgressBar("indexing", "/root", 1, stream=io.StringIO())
    bar.note_gpu(0.5)
    bar.note_gpu(0.25)          # additive: several batches contribute
    bar.note_cpu(9.0)
    bar.note_cpu(10.0)          # absolute: CPU time is a running total
    assert bar.state.gpu_seconds == pytest.approx(0.75)
    assert bar.state.cpu_seconds == pytest.approx(10.0)


def test_in_flight_lists_stay_bounded():
    # A worker that dies mid-file would otherwise leave its path on screen for
    # the rest of the run, and the list would grow without limit.
    bar = ProgressBar("indexing", "/root", 1000, stream=io.StringIO())
    for i in range(500):
        bar.note_decoding(f"/root/{i}.jpg")
    assert len(bar.state.decoding) <= 3


def test_note_finished_removes_the_path():
    bar = ProgressBar("indexing", "/root", 1, stream=io.StringIO())
    bar.note_decoding("/root/a.jpg")
    bar.note_finished("/root/a.jpg")
    assert "/root/a.jpg" not in bar.state.decoding


def test_note_finished_on_an_unknown_path_is_harmless():
    bar = ProgressBar("indexing", "/root", 1, stream=io.StringIO())
    bar.note_finished("/root/never-seen.jpg")  # must not raise
    assert bar.state.decoding == []


def test_context_manager_stops_the_bar_even_on_error():
    stream = _FakeTTY()
    with pytest.raises(RuntimeError):
        with ProgressBar("indexing", "/root", 1, stream=stream):
            raise RuntimeError("boom")
    # A crash mid-scan must not leave the cursor mid-line.
    assert stream.getvalue().endswith("\n")


def test_concurrent_updates_do_not_corrupt_the_state():
    # Worker threads all call into the same state object; the lock is the only
    # thing keeping the counts sane.
    bar = ProgressBar("indexing", "/root", 4000, stream=io.StringIO())

    def worker(n: int) -> None:
        for i in range(100):
            bar.note_completed()
            bar.note_decoding(f"/root/{n}-{i}.jpg")
            bar.note_finished(f"/root/{n}-{i}.jpg")
            bar.note_gpu(0.001)
            bar.set_ring(i, 1000)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert bar.state.done == 800
    assert bar.state.gpu_seconds == pytest.approx(0.8)


def test_stop_is_idempotent():
    stream = _FakeTTY()
    bar = ProgressBar("indexing", "/root", 1, stream=stream).start()
    bar.stop()
    bar.stop()  # a second stop must not double-print or raise