"""Model selection and the front-load/reuse contract."""

import argparse

from media_match.models import ModelCache, ModelNames, resolve_names


class _FakeEncoder:
    model_id = "fake/clip"

    def __init__(self, **kwargs):
        self.loads = 0
        self.kwargs = kwargs

    def ensure_loaded(self):
        self.loads += 1

    def unload(self):
        pass


class _FakeDescriber:
    model_id = "fake/vlm"

    def __init__(self, **kwargs):
        self.loads = 0

    def ensure_loaded(self):
        self.loads += 1

    def unload(self):
        pass


class _FakeDB:
    def __init__(self, clip="", caption=""):
        self.clip = clip
        self.caption = caption

    def get_meta(self, key, default=None):
        if key == "clip_model":
            return self.clip
        if key == "vlm_model":
            return self.caption
        return default


def test_resolve_names_prefers_flag_then_config_then_index(monkeypatch):
    args = argparse.Namespace(clip_model="flag/clip", model=None)
    names = resolve_names(args, {"clip_model": "cfg/clip"},
                          _FakeDB("idx/clip", "idx/vlm"))
    assert names.clip == "flag/clip"
    assert names.caption == "idx/vlm"

    args.clip_model = None
    names = resolve_names(args, {"clip_model": "cfg/clip"},
                          _FakeDB("idx/clip", "idx/vlm"))
    assert names.clip == "cfg/clip"

    names = resolve_names(args, {}, _FakeDB("idx/clip", "idx/vlm"))
    assert names.clip == "idx/clip"
    assert names.caption == "idx/vlm"


def test_cache_loads_and_reuses_clip(monkeypatch, capsys):
    monkeypatch.setattr("media_match.clip.ClipEncoder", _FakeEncoder)
    cache = ModelCache(ModelNames(clip="fake/clip", caption="fake/vlm"))

    first = cache.prepare_clip(enabled=False)
    second = cache.prepare_clip(enabled=False)
    assert first is second
    assert first.loads == 1
    assert "loading fake/clip" in capsys.readouterr().err

    cache.unload_clip()
    third = cache.prepare_clip(enabled=False)
    assert third is not first
    assert third.loads == 1


def test_cache_loads_caption_once(monkeypatch, capsys):
    monkeypatch.setattr("media_match.vision.Describer", _FakeDescriber)
    cache = ModelCache(ModelNames(clip="fake/clip", caption="fake/vlm"))

    first = cache.prepare_caption(enabled=False)
    second = cache.prepare_caption(enabled=False)
    assert first is second
    assert first.loads == 1
    assert "loading fake/vlm" in capsys.readouterr().err


def test_cli_stays_background_free_no_tqdm_monitor_thread():
    # 'media-match' is a foreground CLI: no thread it starts may outlive a
    # command. tqdm (pulled in by the Hugging Face loader) starts a TMonitor
    # daemon thread on the first bar unless monitor_interval is 0; cli.py
    # switches that off at import so model downloads leave nothing behind.
    import threading

    import tqdm

    from media_match import cli  # noqa: F401  (the guard lives at import)

    assert tqdm.tqdm.monitor_interval == 0
    assert [t.name for t in threading.enumerate()] == ["MainThread"]
