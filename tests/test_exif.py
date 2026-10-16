from PIL import Image

from media_match.db import Database
from media_match.exif import read_exif
from media_match.pipeline import ScanJob, prepare, store_result


def _jpeg_with_exif(path, when="2021:07:04 12:30:45", make="Apple",
                    model="iPhone 13"):
    img = Image.new("RGB", (24, 16), (10, 20, 30))
    exif = img.getexif()
    exif[0x010F] = make   # Make
    exif[0x0110] = model  # Model
    exif[36867] = when    # DateTimeOriginal
    img.save(path, exif=exif)


def test_read_exif_extracts_time_and_camera(tmp_path):
    path = tmp_path / "shot.jpg"
    _jpeg_with_exif(path)
    with Image.open(path) as img:
        meta = read_exif(img)
    assert meta["captured_at"] == "2021-07-04T12:30:45"
    assert meta["camera"] == "Apple iPhone 13"


def test_read_exif_handles_no_exif(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(path)
    with Image.open(path) as img:
        meta = read_exif(img)
    assert meta["captured_at"] == ""
    assert meta["camera"] == ""


def test_prepare_carries_exif_into_index(tmp_path):
    path = tmp_path / "shot.jpg"
    _jpeg_with_exif(path)
    job = ScanJob(kind="image", path=str(path), size=path.stat().st_size,
                  mtime=path.stat().st_mtime)
    result = prepare(job, interval=1.0, max_frames=1)
    assert result.captured_at == "2021-07-04T12:30:45"
    assert result.camera == "Apple iPhone 13"

    db = Database(str(tmp_path / "t.db"))
    store_result(db, result)
    rec = db.all_files()[0]
    assert rec.captured_at == "2021-07-04T12:30:45"
    assert rec.camera == "Apple iPhone 13"
    db.close()
