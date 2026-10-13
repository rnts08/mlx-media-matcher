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

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_vlm import load

            self._model, self._processor = load(self.model_id)

    def _ask(self, prompt_text: str, image_path: str, max_tokens: int) -> str:
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
            image=image_path,
            max_tokens=max_tokens,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)

    def describe(self, image_path: str) -> tuple[str, list[str]]:
        """Return (description, keywords) for one image."""
        description = self._ask(_DESC_PROMPT, image_path, max_tokens=60)
        description = description.replace("\n", " ").strip()
        keywords = parse_keywords(self._ask(_KEYWORDS_PROMPT, image_path, max_tokens=40))
        return description, keywords

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except (ImportError, AttributeError):
            pass
