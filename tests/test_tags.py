from media_match.tags import facet_of, normalize_tag, refine


def test_normalize_tag_lowercases_singularizes_and_maps_synonyms():
    assert normalize_tag("Photos of two Men in Grey Suits") == "man gray suit"
    assert normalize_tag("Guy on a Bicycle") == "man bicycle"
    assert normalize_tag("the main things are a dog, grass") == "dog grass"


def test_normalize_tag_handles_irregular_plurals():
    assert normalize_tag("two children") == "child"
    assert normalize_tag("people") == "person"
    assert normalize_tag("geese") == "goose"


def test_normalize_tag_drops_filler_only_tags():
    assert normalize_tag("a photograph of the") == ""
    assert normalize_tag("  ") == ""


def test_facet_of_buckets_tags():
    assert facet_of("woman") == "person"
    assert facet_of("dog") == "animal"
    assert facet_of("bicycle") == "object"
    assert facet_of("beach") == "scene"
    assert facet_of("pizza") == "food"
    assert facet_of("sign") == "text"
    assert facet_of("wedding") == "activity"
    assert facet_of("blue") == "color"
    assert facet_of("nonsense") is None


def test_refine_returns_canonical_tags_and_facets():
    tags, facets = refine(["woman", "red jacket", "beach", "the", "water"], "a person")
    assert tags == ["woman", "red jacket", "beach", "water"]
    assert facets["person"] == ["woman"]
    assert facets["scene"] == ["beach", "water"]


def test_refine_dedupes_and_caps():
    tags, _ = refine(["dog", "dog", "a dog"] + [f"tag{i}" for i in range(40)])
    assert tags.count("dog") == 1
    assert len(tags) <= 24
