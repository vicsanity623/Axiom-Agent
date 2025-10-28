from axiom.dictionary_utils import (
    get_pos_tag_simple,
    get_word_info_from_wordnet,
    lemmatize_word,
)


def test_dictionary_utilities():
    """
    Provides comprehensive coverage for the functions in dictionary_utils.py.
    """
    cat_info = get_word_info_from_wordnet("cat")
    assert cat_info["type"] == "noun"
    assert any("feline" in d for d in cat_info["definitions"])
    assert "feline" in cat_info["related_words"]
    print("get_word_info_from_wordnet: Handled common noun.")

    walk_info = get_word_info_from_wordnet("walk")
    assert walk_info["type"] in {"verb", "noun"}, (
        f"Unexpected type for 'walk': {walk_info['type']}"
    )
    assert any("foot" in d or "move" in d for d in walk_info["definitions"])
    print(f"get_word_info_from_wordnet: Handled '{walk_info['type']}' case for 'walk'.")

    nonsense_info = get_word_info_from_wordnet("flibbertigibbet")
    assert nonsense_info["type"] in {"concept", "noun", "adjective", "verb"}
    print(f"get_word_info_from_wordnet: Handled rare word '{nonsense_info['type']}'.")

    assert get_pos_tag_simple("house") == "noun"
    assert get_pos_tag_simple("jumped") == "verb"
    assert get_pos_tag_simple("beautiful") == "descriptor"
    assert get_pos_tag_simple("quickly") == "adverb"
    print("get_pos_tag_simple: Handled common parts of speech.")

    assert get_pos_tag_simple("axiomatic") in ["descriptor", "noun"]
    print("get_pos_tag_simple: Handled fallback to WordNet.")

    assert get_pos_tag_simple("fnord") == "concept"
    print("get_pos_tag_simple: Handled non-existent word.")

    assert lemmatize_word("cats") == "cat"
    assert lemmatize_word("geese") == "goose"

    assert lemmatize_word("running", pos="v") == "run"
    assert lemmatize_word("was", pos="v") == "be"

    assert lemmatize_word("better", pos="a") == "good"  # 'a' for adjective
    print("lemmatize_word: Handled various forms and parts of speech.")
