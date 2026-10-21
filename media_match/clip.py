"""CLIP image/text embeddings on MLX.

Uses the vendored pure-MLX CLIP implementation (see clip_model.py) with
mlx-community checkpoints. CLIP gives a single 512-dim vector space shared by
images and text, so:
  - text->image search  = rank by cosine similarity of query text embedding
                          against stored image embeddings
  - image->image search = rank by cosine similarity between image embeddings
Both are one forward pass per input (no autoregressive generation).
"""

from __future__ import annotations

import numpy as np

DEFAULT_CLIP_MODEL = "mlx-community/clip-vit-base-patch32"


class ClipEncoder:
    def __init__(self, model_id: str = DEFAULT_CLIP_MODEL, batch_size: int = 16):
        self.model_id = model_id
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def ensure_loaded(self) -> None:
        """Load the weights now, so callers can announce the pause.

        The scan calls this before it starts its progress bar: the first-run
        download otherwise happens inside the first GPU batch, where its
        progress bars fight the scan's bar and the weight load looks like a
        hang.
        """
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import snapshot_download

        from .clip_model import CLIPModel
        from .clip_tokenizer import CLIPTokenizer

        path = snapshot_download(self.model_id)
        self._model = CLIPModel.from_pretrained(path)
        self._tokenizer = CLIPTokenizer.from_pretrained(path)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """(N, 512) L2-normalized text embeddings."""
        import mlx.core as mx

        self._ensure_loaded()
        tokens = self._tokenizer.tokenize(texts)
        ids = self._tokenizer.pad(tokens)
        feats = self._model.get_text_features(ids)
        feats = feats / mx.linalg.norm(feats, axis=-1, keepdims=True)
        return np.asarray(feats, dtype=np.float32)

    def embed_images(self, images: list["np.ndarray"]) -> np.ndarray:
        """(N, 512) L2-normalized image embeddings from preprocessed arrays."""
        import mlx.core as mx

        self._ensure_loaded()
        out = []
        for start in range(0, len(images), self.batch_size):
            batch = mx.array(np.stack(images[start : start + self.batch_size]))
            feats = self._model.get_image_features(batch)
            feats = feats / mx.linalg.norm(feats, axis=-1, keepdims=True)
            out.append(np.asarray(feats, dtype=np.float32))
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)

    def embed_image_paths(self, paths: list[str]) -> np.ndarray:
        """Load, preprocess and embed image files."""
        from PIL import Image

        from .clip_processor import process_image

        self._ensure_loaded()
        processed = []
        for path in paths:
            with Image.open(path) as img:
                processed.append(process_image(img.convert("RGB")))
        return self.embed_images(processed)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import mlx.core as mx

            (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()
        except (ImportError, AttributeError):
            pass
