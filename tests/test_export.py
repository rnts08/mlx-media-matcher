import json

from media_match import cli
from media_match.db import Database


def _build(root):
    root.mkdir(exist_ok=True)
    db = Database(str(root / ".media-match.db"))
    db.upsert_file(path=str(root / "a.jpg"), size=1000, mtime=1.0, sha1="a",
                   width=100, height=80, phash="0" * 16, description="a dog",
                   keywords="dog", captured_at="2021-07-04T12:00:00",
                   camera="Apple iPhone 13")
    db.upsert_file(path=str(root / "b.jpg"), size=500, mtime=2.0, sha1="b",
                   width=50, height=40, phash="1" * 16, description="a cat",
                   keywords="cat")
    a_id = db.get_file(str(root / "a.jpg"))["id"]
    b_id = db.get_file(str(root / "b.jpg"))["id"]
    db.apply_refinement("image", [
        (a_id, "dog", "dog", "a dog", [("dog", "animal")]),
        (b_id, "cat", "cat", "a cat", [("cat", "animal")]),
    ])
    db.set_types([(a_id, "animal", 0.9)])
    db.set_clusters("image", {1: [a_id]}, {1: "Alice"})
    db.upsert_video(path=str(root / "v.mp4"), size=2000, mtime=3.0, sha1="v",
                    duration=5.0, width=640, height=480, fps=30.0, codec="h264",
                    bitrate=1000, signature="")
    db.close()
    return root


def test_export_json(tmp_path, capsys):
    root = _build(tmp_path)
    assert cli.main(["export", str(root), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 3
    a = next(r for r in data if r["path"].endswith("a.jpg"))
    assert a["captured_at"] == "2021-07-04T12:00:00"
    assert a["camera"] == "Apple iPhone 13"
    assert a["type"] == "animal"
    assert a["cluster"] == "Alice"
    assert a["tags"] == "dog"
    v = next(r for r in data if r["path"].endswith("v.mp4"))
    assert v["kind"] == "video" and v["duration"] == 5.0


def test_export_csv(tmp_path, capsys):
    root = _build(tmp_path)
    assert cli.main(["export", str(root), "--format", "csv"]) == 0
    out = capsys.readouterr().out
    assert ("kind,path,width,height,size,captured_at,camera,description,"
            "keywords,type,tags,cluster,duration") in out
    assert "a.jpg" in out and "v.mp4" in out


def test_export_markdown(tmp_path, capsys):
    root = _build(tmp_path)
    assert cli.main(["export", str(root), "--format", "md"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| path | kind |")
    assert "a.jpg" in out


def test_export_out_file(tmp_path):
    root = _build(tmp_path)
    dest = tmp_path / "index.json"
    assert cli.main(["export", str(root), "--format", "json",
                     "--out", str(dest)]) == 0
    assert len(json.loads(dest.read_text())) == 3
