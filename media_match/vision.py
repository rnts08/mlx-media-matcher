"""Vision-model based image description using a local MLX model.

Default model: mlx-community/SmolVLM-256M-Instruct-4bit — the smallest
practically usable vision model on MLX (~150 MB, roughly 1s per image on
Apple Silicon). Override with --model for better accuracy
(e.g. mlx-community/Qwen2-VL-2B-Instruct-4bit).

Each image gets two short passes (the small model follows short, single-task
prompts much better than compound ones):
  1. one-sentence description
  2. list of the main people/animals/objects, parsed into keywords
"""

from __future__ import annotations

import re

DEFAULT_MODEL = "mlx-community/SmolVLM-256M-Instruct-4bit"

_DESC_PROMPT = "Describe this photo in one sentence."
_KEYWORDS_PROMPT = "What are the main things in this photo? Reply only with comma-separated words."

# leading boilerplate the small model likes to add before the actual keyword list
_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?main (?:things|objects|elements|items) (?:in|of) (?:this|the) "
    r"(?:photo|image|picture) (?:are|is)\s*",
    re.IGNORECASE,
)

# words that are useless as search keywords
_FILLER = {
    "a", "an", "the", "in", "of", "on", "at", "to", "is", "are", "this", "that",
    "it", "there", "with", "and", "some", "can", "see", "we", "main", "things",
    "photo", "picture", "image", "shows", "visible", "foreground", "background",
}


def parse_keywords(text: str) -> list[str]:
    """Turn the model's keyword answer into clean search keywords."""
    text = _PREFIX_RE.sub("", text.strip())
    text = re.sub(r"[\n.]", ",", text)
    parts = re.split(r",|\band\b", text)
    keywords: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = part.strip().strip(".").lower()
        words = part.split()
        if not words or all(w in _FILLER for w in words):
            continue
        if part in seen or len(part) > 32:
            continue
        seen.add(part)
        keywords.append(part)
    return keywords[:12]


class Describer:
    """Loads the MLX vision model once and describes images with it."""

    def __init__(self, model_id: str | None = DEFAULT_MODEL):
        # a None --model (flag not given) must not reach mlx_vlm.load, which
        # fails on Path(None); fall back to the default instead
        self.model_id = model_id or DEFAULT_MODEL
        self._model = None
        self._processor = None

    def ensure_loaded(self) -> None:
        """Load the weights now, so callers can announce the pause."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_vlm import load

            self._model, self._processor = load(self.model_id)

    def _ask(self, prompt_text: str, image, max_tokens: int) -> str:
        """One prompt against one image (a path or an already-decoded PIL image)."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_loaded()
        prompt = apply_chat_template(
            self._processor, self._model.config, prompt_text, num_images=1
        )
        result = generate(
            self._model,
            self._processor,
            prompt,
            image=image,
            max_tokens=max_tokens,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)

    def describe(self, image) -> tuple[str, list[str]]:
        """Return (description, keywords) for one image (path or PIL image)."""
        description = self._ask(_DESC_PROMPT, image, max_tokens=60)
        description = description.replace("\n", " ").strip()
        keywords = parse_keywords(self._ask(_KEYWORDS_PROMPT, image, max_tokens=40))
        return description, keywords

    def describe_batch(self, images: list) -> list[tuple[str, list[str]]]:
        """Caption a batch of images in fewer VLM passes.

        Sends all images per prompt (description pass, then keywords pass)
        instead of one call per image. Returns one (description, keywords)
        tuple per input image, in order.
        """
        if not images:
            return []
        descriptions = self._ask_batch(_DESC_PROMPT, images, max_tokens=60)
        descriptions = [d.replace("\n", " ").strip() for d in descriptions]
        raw_keywords = self._ask_batch(_KEYWORDS_PROMPT, images, max_tokens=40)
        return [(d, parse_keywords(k)) for d, k in zip(descriptions, raw_keywords)]

    def _ask_batch(self, prompt_text: str, images: list, max_tokens: int) -> list[str]:
        """One prompt against a batch of images; returns one string per image."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_loaded()
        # one prompt per image, batched together
        prompts = [
            apply_chat_template(self._processor, self._model.config, prompt_text,
                                num_images=1)
            for _ in images
        ]
        result = generate(
            self._model,
            self._processor,
            prompts,
            image=images,
            max_tokens=max_tokens,
            verbose=False,
        )
        if isinstance(result, list):
            return [r.text if hasattr(r, "text") else str(r) for r in result]
        # single GenerationResult or list of them
        text = result.text if hasattr(result, "text") else str(result)
        return [text] * len(images)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import mlx.core as mx

            (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()
        except (ImportError, AttributeError):
            pass
