"""Diagnostics: Pillow's warnings must never reach the user as noise.

A warning that prints its own site-packages path and no filename is useless in
a scan of thousands of files. These tests pin two things: the noise is gone,
and the information survives as something you can act on.
"""

import warnings

import pytest
from PIL import Image

from media_match.diagnose import (
    CORRUPT,
    NOTABLE,
    OK,
    TRUNCATED,
    classify,
    diagnose_image,
)


def _write_image(path, size=(64, 48), color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _truncate(path, drop=225):
    path.write_bytes(path.read_bytes()[:-drop])


# -- no warning escapes ---------------------------------------------------------


def test_diagnose_image_emits_no_warnings_on_a_truncated_file(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p, size=(400, 300))
    _truncate(p, drop=225)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = diagnose_image(str(p))
    assert found.status == TRUNCATED
    assert caught == [], f"warnings leaked to the user: {[str(w.message) for w in caught]}"


def test_diagnose_image_emits_no_warnings_on_a_malformed_mpo(tmp_path):
    # Pillow warns "malformed MPO file, it will be interpreted as a base JPEG".
    p = tmp_path / "a.jpg"
    _write_image(p)
    data = p.read_bytes()
    p.write_bytes(data[:2] + b"\xff\xd8\xff\xe1" + data[2:])  # bogus APP1 segment

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnose_image(str(p))
    assert not [w for w in caught if "PIL" in str(w.message)], \
        f"Pillow warnings leaked: {[str(w.message) for w in caught]}"


def test_scan_tolerates_truncation_without_printing(capsys, tmp_path):
    from media_match import cli

    p = tmp_path / "a.jpg"
    _write_image(p, size=(400, 300))
    _truncate(p, drop=225)

    class FakeEncoder:
        def embed_images(self, arrays):
            import numpy as np

            return np.stack([np.ones(4, dtype=np.float32) for _ in arrays])

        def unload(self):
            pass

    from media_match.pipeline import build_jobs, run_staged_scan, store_result
    from media_match.db import Database

    db = Database(str(tmp_path / "t.db"))
    jobs, *_ = build_jobs(str(tmp_path), db, include_video=False, reanalyze=False)
    assert jobs, "the truncated file should still be indexable"
    seen = []
    run_staged_scan(jobs, FakeEncoder(), interval=1.0, max_frames=10,
                    batch_size=2, workers=1, on_result=seen.append)
    for result in seen:
        store_result(db, result)

    captured = capsys.readouterr()
    assert db.count_files() == 1
    assert "site-packages" not in captured.err
    assert "PIL" not in captured.err


# -- the information survives ---------------------------------------------------


def test_truncated_file_names_itself_and_the_fix(tmp_path):
    p = tmp_path / "IMG_0042.jpg"
    _write_image(p, size=(400, 300))
    _truncate(p, drop=225)
    found = diagnose_image(str(p))
    assert found.status == TRUNCATED
    assert "400x300" in found.message       # what the header claims
    assert "media-match check" in found.message  # what to do about it
    assert found.actionable is True


def test_clean_file_has_no_message_to_print(tmp_path):
    p = tmp_path / "a.jpg"
    _write_image(p)
    found = diagnose_image(str(p))
    assert found == type(found)(OK, "")
    assert found.ok and not found.actionable


def test_corrupt_file_says_so(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"not an image at all")
    found = diagnose_image(str(p))
    assert found.status == CORRUPT
    assert "cannot" in found.message
    assert found.actionable is False  # nothing to repair, only quarantine


def test_notable_is_distinct_from_corrupt(tmp_path):
    # A malformed MPO still decodes: reporting it as 'corrupt' would send the
    # user to quarantine a perfectly good photo.
    assert classify(["Image appears to be a malformed MPO file"]).status == NOTABLE


@pytest.fixture
def fake_encoder(monkeypatch):
    """Stand in for ClipEncoder so CLI scan tests need no model download."""
    import numpy as np

    from media_match import cli

    class FakeEncoder:
        def embed_images(self, arrays):
            return np.stack([np.ones(4, dtype=np.float32) for _ in arrays])

        def unload(self):
            pass

    monkeypatch.setattr("media_match.clip.ClipEncoder", lambda **kw: FakeEncoder())
    return FakeEncoder


def test_scan_reports_the_file_not_a_traceback(capsys, tmp_path, fake_encoder):
    """A file the scan cannot read must be named, with a usable explanation."""
    from media_match import cli

    _write_image(tmp_path / "ok.jpg")
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image at all")

    assert cli.main(["scan", str(tmp_path)]) == 0
    out = capsys.readouterr()
    combined = out.out + out.err
    # the filename is there, and so is something better than a traceback
    assert "broken.jpg" in combined
    assert "UnidentifiedImageError" not in combined
    assert "Traceback" not in combined
    assert "media-match check" in combined
    assert "ok.jpg" not in combined.split("could not be indexed")[-1]


def test_scan_does_not_report_a_file_it_indexed(capsys, tmp_path, fake_encoder):
    """A healthy file that was indexed must not appear in the failure report."""
    from media_match import cli

    _write_image(tmp_path / "clean.jpg")

    assert cli.main(["scan", str(tmp_path)]) == 0
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert "could not be indexed" not in combined


def test_decompression_bomb_is_reported_in_megapixels():
    msg = ("Image size (108000000 pixels) exceeds limit of 89478485 pixels, "
           "could be decompression bomb DOS attack.")
    found = classify([msg])
    assert found.status == NOTABLE
    assert "108 MP" in found.message
    assert "nothing is damaged" in found.message
    assert found.actionable is False


def test_bomb_and_mpo_win_over_truncation_in_the_message():
    # A truncated file can trip several warnings; the more specific one is
    # the more useful thing to say. Both matchers must see the real Pillow
    # text, so the bomb message keeps its "decompression bomb" tail.
    found = classify([
        "Truncated File Read",
        "Image size (108000000 pixels) exceeds limit of 89478485 pixels, "
        "could be decompression bomb DOS attack.",
    ])
    assert found.status == NOTABLE
    assert "108 MP" in found.message


def test_classify_ignores_unrelated_warnings():
    assert classify(["some other warning entirely"]).status == OK


def test_message_names_the_dimensions_when_it_has_them():
    found = classify(["Truncated File Read"], width=400, height=300)
    assert "400x300 header intact" in found.message


def test_message_degrades_gracefully_without_dimensions():
    found = classify(["Truncated File Read"], width=0, height=0)
    assert found.status == TRUNCATED
    assert "header intact" not in found.message
    assert "ends early" in found.message
    assert "media-match check" in found.message