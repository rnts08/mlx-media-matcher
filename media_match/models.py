"""Model selection and lifetime for one command run.

Every stage that needs a model asks this module for it. The important rule is
ordering: the model is loaded *before* the stage creates its progress bar, so a
first-run download or MLX kernel compile cannot appear in the middle of the
display.  A cache also keeps the CLIP encoder alive between stages that use it
without a caption pass in between.  ``analyze`` deliberately releases CLIP
while the caption model runs, then reloads it before the next bar.

The imports are deliberately inside methods so the CLI and tests can replace
``media_match.clip.ClipEncoder`` / ``media_match.vision.Describer`` with fakes
without this module holding the real class from import time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clip import DEFAULT_CLIP_MODEL
from .console import load_model
from .vision import DEFAULT_MODEL


@dataclass(frozen=True)
class ModelNames:
    """Resolved model ids for one run (flag > config > index > default)."""

    clip: str = DEFAULT_CLIP_MODEL
    caption: str = DEFAULT_MODEL


def resolve_names(args=None, cfg: dict | None = None, db=None) -> ModelNames:
    """Resolve both model families without touching the network.

    An explicit command-line flag wins, then the config file, then the model
    recorded in the index (so search/classify do not silently switch spaces on
    an index built with a custom ``--clip-model``), then the built-in default.
    """
    cfg = cfg or {}
    index_clip = ""
    index_caption = ""
    if db is not None:
        try:
            index_clip = db.get_meta("clip_model", "") or ""
            index_caption = db.get_meta("vlm_model", "") or ""
        except Exception:  # noqa: BLE001 - an old/broken index must not abort
            index_clip = index_caption = ""

    clip = (getattr(args, "clip_model", None) or cfg.get("clip_model")
            or index_clip or DEFAULT_CLIP_MODEL)
    caption = (getattr(args, "model", None) or cfg.get("model")
               or index_caption or DEFAULT_MODEL)
    return ModelNames(clip=clip, caption=caption)


class ModelCache:
    """Lazily construct, preload and reuse the models a command needs."""

    def __init__(self, names: ModelNames, *, batch_size: int = 16):
        self.names = names
        self.batch_size = max(1, int(batch_size or 1))
        self._clip = None
        self._caption = None
        self._clip_ready = False
        self._caption_ready = False

    def clip(self):
        if self._clip is None:
            from .clip import ClipEncoder

            self._clip = ClipEncoder(model_id=self.names.clip,
                                     batch_size=self.batch_size)
        return self._clip

    def caption(self):
        if self._caption is None:
            from .vision import Describer

            self._caption = Describer(model_id=self.names.caption)
        return self._caption

    def prepare_clip(self, *, enabled: bool | None = None):
        """Load CLIP now and return the encoder; reuse an already-loaded one."""
        encoder = self.clip()
        if not self._clip_ready:
            load_model(encoder, "CLIP model", enabled=enabled)
            self._clip_ready = True
        return encoder

    def prepare_caption(self, *, enabled: bool | None = None):
        """Load the caption model now and return it."""
        describer = self.caption()
        if not self._caption_ready:
            load_model(describer, "caption model", enabled=enabled)
            self._caption_ready = True
        return describer

    def unload_clip(self) -> None:
        if self._clip is not None:
            unload = getattr(self._clip, "unload", None)
            if callable(unload):
                unload()
            _mark_released(self._clip)
        self._clip = None
        self._clip_ready = False

    def unload_caption(self) -> None:
        if self._caption is not None:
            unload = getattr(self._caption, "unload", None)
            if callable(unload):
                unload()
            _mark_released(self._caption)
        self._caption = None
        self._caption_ready = False

    def unload(self) -> None:
        self.unload_clip()
        self.unload_caption()


def _mark_released(model_holder) -> None:
    try:
        setattr(model_holder, "_media_match_model_ready", False)
    except (AttributeError, TypeError):  # pragma: no cover - exotic stand-in
        pass
