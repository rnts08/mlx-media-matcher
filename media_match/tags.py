"""Tag refinement: turn loose VLM keywords into canonical tags + facets.

The vision model answers with free text ("a man", "the main things are a dog,
grass"). This module lowercases, singularizes, drops filler, maps obvious
synonyms to one canonical tag, and then buckets every tag into a coarse
*facet* (person / animal / object / scene / food / plant / text / activity /
color). Facets let the library be browsed by kind as well as by word, and are
what the people clustering and the type classifier build on.
"""

from __future__ import annotations

import re

FACETS = (
    "person", "animal", "object", "scene", "food",
    "plant", "text", "activity", "color",
)

# words that carry no search value on their own
FILLER = {
    "a", "an", "the", "in", "of", "on", "at", "to", "is", "are", "this", "that",
    "it", "there", "with", "and", "some", "can", "see", "we", "main", "things",
    "thing", "object", "objects", "elements", "items", "photo", "photograph",
    "picture", "image", "images", "shows", "showing", "visible", "foreground",
    "background", "large", "small", "few", "several", "many", "one", "two",
    "front", "scene", "view", "area", "part", "piece", "kind", "type",
}

# irregular plurals that a suffix rule cannot fix
_IRREGULAR = {
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
    "oxen": "ox", "knives": "knife", "leaves": "leaf", "wolves": "wolf",
    "halves": "half", "loaves": "loaf", "shelves": "shelf",
}

# obvious synonyms -> one canonical tag (keys are already singularized)
_SYNONYMS = {
    "guy": "man", "gentleman": "man", "male": "man",
    "lady": "woman", "female": "woman", "girl": "girl",
    "boy": "boy", "kid": "child", "youngster": "child", "toddler": "child",
    "individual": "person", "human": "person",
    "human being": "person", "crowd": "crowd", "portrait": "portrait",
    "puppy": "dog", "kitten": "cat", "cellphone": "phone", "mobile": "phone",
    "bike": "bicycle", "automobile": "car", "auto": "car", "tv": "television",
    "pc": "laptop", "notebook": "laptop", "sneaker": "shoe", "spectacles": "glasses",
    "eyeglasses": "glasses", "sea": "ocean", "grey": "gray",
}

# canonical tag (or any word within a multi-word tag) -> facet
_LEXICON: dict[str, set[str]] = {
    "person": {
        "person", "man", "woman", "boy", "girl", "child", "baby", "baby",
        "people", "face", "selfie", "portrait", "groom", "bride", "teenager",
        "adult", "couple", "family", "friend", "crowd", "smile",
    },
    "animal": {
        "dog", "cat", "puppy", "kitten", "bird", "horse", "cow", "sheep",
        "elephant", "fish", "bear", "lion", "tiger", "rabbit", "animal", "pet",
        "duck", "goose", "monkey", "deer", "fox", "squirrel",
    },
    "object": {
        "car", "bicycle", "motorcycle", "phone", "computer", "laptop", "table",
        "chair", "bottle", "cup", "glass", "book", "bag", "shoe", "hat",
        "watch", "camera", "guitar", "ball", "umbrella", "key", "clock", "vase",
        "tool", "furniture", "window", "door", "toy", "sign", "box", "can",
        "jacket", "shirt", "coat", "dress", "pant", "jean", "sweater", "tie",
        "scarf", "glove", "sock", "uniform", "clothing", "backpack", "basket",
    },
    "scene": {
        "beach", "ocean", "mountain", "forest", "tree", "sky", "cloud",
        "sunset", "sunrise", "snow", "grass", "field", "river", "lake",
        "waterfall", "desert", "city", "street", "building", "house", "room",
        "garden", "park", "night", "landscape", "scenery", "skyline", "road",
        "bridge", "coast", "valley", "hill", "island", "water", "ceiling",
    },
    "food": {
        "pizza", "cake", "bread", "fruit", "apple", "banana", "orange",
        "coffee", "tea", "wine", "beer", "plate", "meal", "food", "dessert",
        "salad", "pasta", "sushi", "burger", "cheese", "rice", "soup",
    },
    "plant": {
        "flower", "rose", "plant", "leaf", "branch", "garden", "blossom",
        "bush", "vegetable",
    },
    "text": {
        "text", "sign", "document", "letter", "paper", "screenshot", "page",
        "logo", "number", "word", "writing", "sentence",
    },
    "activity": {
        "running", "walking", "sitting", "standing", "playing", "swimming",
        "dancing", "eating", "drinking", "wedding", "party", "birthday",
        "sport", "soccer", "football", "basketball", "concert", "cooking",
        "reading", "working", "riding", "hiking", "surfing", "sleeping",
    },
    "color": {
        "red", "blue", "green", "yellow", "black", "white", "orange", "pink",
        "purple", "brown", "gray", "gold", "silver", "beige", "turquoise",
    },
}

# longest facets win ties so "wedding dress" reads as an activity, not object
_FACET_ORDER = (
    "person", "activity", "animal", "food", "plant", "text",
    "scene", "object", "color",
)

_WORD_RE = re.compile(r"[^\w\s-]")


def singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def normalize_tag(text: str) -> str:
    """Lowercase, strip punctuation, singularize, map synonyms. '' if filler."""
    text = _WORD_RE.sub(" ", text.lower().strip())
    words = []
    for raw in text.split():
        if raw in FILLER:
            continue
        word = _IRREGULAR.get(raw) or singularize(raw)
        word = _SYNONYMS.get(word, word)
        if word and word not in FILLER:
            words.append(word)
    return " ".join(words)


def facet_of(tag: str) -> str | None:
    """Which facet a (possibly multi-word) canonical tag belongs to."""
    words = tag.split()
    for facet in _FACET_ORDER:
        lexicon = _LEXICON[facet]
        if tag in lexicon or any(w in lexicon for w in words):
            return facet
    return None


def refine(keywords: list[str], description: str = "") -> tuple[list[str], dict[str, list[str]]]:
    """Raw keywords -> (canonical tags, facet -> tags).

    Tags keep their source order, are de-duplicated, and are capped so one
    verbose caption cannot dominate the tag cloud.
    """
    tags: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        tag = normalize_tag(raw)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= 24:
            break

    facets: dict[str, list[str]] = {}
    for tag in tags:
        facet = facet_of(tag)
        if facet:
            facets.setdefault(facet, []).append(tag)
    return tags, facets
