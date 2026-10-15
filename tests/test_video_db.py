from media_match.db import Database


def _add_video(db, path="/v/a.mp4", sha1="aa", phashes=("00", "01")):
    vid = db.upsert_video(
        path=path, size=100, mtime=1.0, sha1=sha1, duration=2.0,
        width=320, height=240, fps=10.0, codec="h264", bitrate=500_000,
        signature=min(phashes), description="", keywords="",
    )
    frames = [
        (i, i * 1.0, 320, 240, ph, None) for i, ph in enumerate(phashes)
    ]
    db.set_video_frames(vid, frames)
    return vid


def test_video_roundtrip(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    vid = _add_video(db)
    v = db.get_video("/v/a.mp4")
    assert v["duration"] == 2.0 and v["bitrate"] == 500_000
    assert v["frame_count"] == 2
    frames = db.video_frames(vid)
    assert [f["t"] for f in frames] == [0.0, 1.0]
    assert frames[0]["phash"] == "00"


def test_all_videos_with_frames_attaches_phashes(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _add_video(db, phashes=("0a", "0b", "0c"))
    records = db.all_videos_with_frames()
    assert len(records) == 1
    assert records[0].phashes == ["0a", "0b", "0c"]
    assert records[0].phash == "0a"  # duck-typed signature


def test_video_embeddings(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    vid = _add_video(db)
    db.set_video_frames(vid, [(0, 0.0, 320, 240, "00", b"\x00" * 8),
                              (1, 1.0, 320, 240, "01", None)])
    embeddings = db.video_embeddings()
    assert list(embeddings) == [vid]
    assert len(embeddings[vid]) == 1  # only the embedded frame


def test_delete_videos_removes_frames(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    vid = _add_video(db)
    db.delete_videos([vid])
    assert db.get_video("/v/a.mp4") is None
    assert db.video_frames(vid) == []
    assert db.count_videos() == 0


def test_delete_media_routes_by_type(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.upsert_file(path="/i/a.jpg", size=1, mtime=0, sha1="f", width=2,
                   height=2, phash="00", description="", keywords="")
    _add_video(db)
    file_rec = db.all_files()[0]
    video_rec = db.all_videos()[0]
    # ids collide across tables on purpose: routing must keep them apart
    assert file_rec.id == video_rec.id == 1
    db.delete_media([video_rec])
    assert db.get_video("/v/a.mp4") is None
    assert db.get_file("/i/a.jpg") is not None
    db.delete_media([file_rec])
    assert db.get_file("/i/a.jpg") is None


def test_search_videos_fts(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    vid = db.upsert_video(path="/v/a.mp4", size=1, mtime=0, sha1="a",
                          duration=1, width=2, height=2, fps=1, codec="h264",
                          bitrate=1, signature="00",
                          description="a surfer riding a wave",
                          keywords="surfer, wave, ocean")
    assert [r["id"] for r in db.search_videos(["surfer"], limit=5)] == [vid]
    assert db.search_videos(["nonexistent"], limit=5) == []


def test_keyword_counts_include_videos(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.upsert_video(path="/v/a.mp4", size=1, mtime=0, sha1="a", duration=1,
                    width=2, height=2, fps=1, codec="h264", bitrate=1,
                    signature="00", keywords="ocean, wave")
    assert ("ocean", 1) in db.keyword_counts(include_videos=True)
    assert db.keyword_counts(include_videos=False) == []
