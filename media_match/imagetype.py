"""Zero-shot photo *type* classification with CLIP.

Instead of training a classifier, each category is described by a small
ensemble of prompt sentences ("a selfie of a person", ...). The text encoder
turns them into vectors in the same space as the image embeddings, the prompts
of one category are averaged into a single class vector, and an image is
labeled with the category whose vector it is closest to (cosine). No extra
model, no labels, no training run — just the CLIP checkpoint the index already
uses.

This is a coarse taxonomy (selfie vs. scene vs. object), which is what CLIP is
good at; it is not a fine-grained classifier.
"""

from __future__ import annotations

import numpy as np

# category -> prompt ensemble. More prompts per category = steadier centroid.
CATEGORY_PROMPTS: dict[str, list[str]] = {
    "selfie": [
        "a selfie of a person holding the camera at arm's length",
        "a close-up self-portrait of a person's face",
    ],
    "portrait": [
        "a portrait photo of a person posing for the camera",
        "a head-and-shoulders photo of a single person",
    ],
    "group": [
        "a group photo of several people together",
        "a photo of a group of friends smiling at the camera",
    ],
    "scene": [
        "a landscape photograph of nature",
        "a scenic outdoor view of mountains, sea or sky",
    ],
    "city": [
        "a photograph of a city street or buildings",
        "an urban skyline or street scene",
    ],
    "animal": [
        "a photograph of an animal",
        "a close-up photo of a pet or wild animal",
    ],
    "object": [
        "a close-up photo of a single object on a plain background",
        "a product photo of one item",
    ],
    "food": [
        "a photograph of food on a plate",
        "a close-up of a meal or drink",
    ],
    "document": [
        "a scan or photograph of a document or page of text",
        "a screenshot of a computer screen or app",
    ],
    "art": [
        "a painting or drawing",
        "a piece of artwork or illustration",
    ],
}

DEFAULT_TYPE = "other"


def category_prompts() -> list[tuple[str, str]]:
    """(category, prompt) pairs in a stable order."""
    return [(cat, prompt) for cat, prompts in CATEGORY_PROMPTS.items() for prompt in prompts]


class TypeClassifier:
    """Classify image embeddings against the prompt-ensembled categories."""

    def __init__(self, encoder):
        self.encoder = encoder
        self._names: list[str] = []
        self._centroids: np.ndarray | None = None

    def _ensure(self) -> None:
        if self._centroids is not None:
            return
        pairs = category_prompts()
        vectors = self.encoder.embed_texts([prompt for _cat, prompt in pairs])
        names = list(CATEGORY_PROMPTS)
        centroids = []
        for name in names:
            idx = [i for i, (cat, _p) in enumerate(pairs) if cat == name]
            mean = vectors[idx].mean(axis=0)
            norm = float(np.linalg.norm(mean)) or 1.0
            centroids.append(mean / norm)
        self._names = names
        self._centroids = np.stack(centroids)

    def classify(self, vectors: np.ndarray) -> list[tuple[str, float]]:
        """(N, 512) image vectors -> (category, top_score) per row.

        Scores are cosine similarities; the runner-up is used to fall back to
        ``other`` when the top category has no real margin over the second.
        """
        if len(vectors) == 0:
            return []
        self._ensure()
        assert self._centroids is not None
        sims = vectors @ self._centroids.T  # (N, C)
        out: list[tuple[str, float]] = []
        for row in sims:
            order = np.argsort(-row)
            top = int(order[0])
            score = float(row[top])
            margin = score - float(row[order[1]]) if len(order) > 1 else score
            if margin < 0.01:  # ambiguous between two categories
                out.append((DEFAULT_TYPE, score))
            else:
                out.append((self._names[top], score))
        return out
