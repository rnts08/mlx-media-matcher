import numpy as np

from media_match.imagetype import (
    CATEGORY_PROMPTS,
    DEFAULT_TYPE,
    TypeClassifier,
    category_prompts,
)


class FakeTextEncoder:
    """Maps every prompt to a one-hot vector for its category (no MLX)."""

    def __init__(self):
        self.index = {name: i for i, name in enumerate(CATEGORY_PROMPTS)}

    def embed_texts(self, texts):
        out = np.zeros((len(texts), len(CATEGORY_PROMPTS)), dtype=np.float32)
        prompt_to_cat = {
            prompt: cat for cat, prompts in CATEGORY_PROMPTS.items() for prompt in prompts
        }
        for row, text in enumerate(texts):
            out[row, self.index[prompt_to_cat[text]]] = 1.0
        return out


def test_category_prompts_cover_every_category():
    pairs = category_prompts()
    assert {cat for cat, _ in pairs} == set(CATEGORY_PROMPTS)
    assert all(prompt.strip() for _cat, prompt in pairs)


def test_classify_picks_the_matching_category():
    classifier = TypeClassifier(FakeTextEncoder())
    names = list(CATEGORY_PROMPTS)
    vectors = np.zeros((len(names), len(names)), dtype=np.float32)
    for i in range(len(names)):
        vectors[i, i] = 1.0
    results = classifier.classify(vectors)
    assert [name for name, _score in results] == names
    assert all(score == 1.0 for _name, score in results)


def test_classify_falls_back_to_other_when_ambiguous():
    classifier = TypeClassifier(FakeTextEncoder())
    n = len(CATEGORY_PROMPTS)
    vector = np.zeros((1, n), dtype=np.float32)
    vector[0, 0] = 0.5
    vector[0, 1] = 0.5  # no margin between two categories
    (name, _score), = classifier.classify(vector)
    assert name == DEFAULT_TYPE


def test_classify_empty_batch():
    assert TypeClassifier(FakeTextEncoder()).classify(np.zeros((0, 4), dtype=np.float32)) == []
