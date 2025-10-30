from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from nltk.corpus import wordnet as wn

from axiom import dictionary_utils

# ---------------------------------------------------------------------
# _ensure_nltk_data_downloaded
# ---------------------------------------------------------------------


def test_ensure_nltk_data_downloaded_all_present(monkeypatch):
    """nltk.download should NOT be called when corpora already exist."""
    monkeypatch.setattr(wn, "synsets", lambda word: True)
    monkeypatch.setattr(dictionary_utils.nltk, "pos_tag", lambda words: True)

    download_spy = MagicMock()
    monkeypatch.setattr(dictionary_utils.nltk, "download", download_spy)

    importlib.reload(dictionary_utils)
    download_spy.assert_not_called()


def test_ensure_nltk_data_downloaded_wordnet_missing(monkeypatch):
    """nltk.download('wordnet') should be called if wordnet is missing."""
    monkeypatch.setattr(wn, "synsets", MagicMock(side_effect=LookupError))
    monkeypatch.setattr(dictionary_utils.nltk, "pos_tag", lambda words: True)

    download_spy = MagicMock()
    monkeypatch.setattr(dictionary_utils.nltk, "download", download_spy)

    importlib.reload(dictionary_utils)
    download_spy.assert_any_call("wordnet")


def test_ensure_nltk_data_downloaded_tagger_missing(monkeypatch):
    """nltk.download('averaged_perceptron_tagger') should be called if tagger missing."""
    monkeypatch.setattr(wn, "synsets", lambda word: True)
    monkeypatch.setattr(
        dictionary_utils.nltk, "pos_tag", MagicMock(side_effect=LookupError)
    )

    download_spy = MagicMock()
    monkeypatch.setattr(dictionary_utils.nltk, "download", download_spy)

    importlib.reload(dictionary_utils)
    download_spy.assert_any_call("averaged_perceptron_tagger")


# ---------------------------------------------------------------------
# get_word_info_from_wordnet
# ---------------------------------------------------------------------


def test_get_word_info_from_wordnet_not_found():
    """Unknown words should return an empty list."""
    assert dictionary_utils.get_word_info_from_wordnet("flibbertigibbet") == []


def test_get_word_info_from_wordnet_polysemy():
    """Word with multiple POS (e.g., 'run')."""
    results = dictionary_utils.get_word_info_from_wordnet("run")
    assert len(results) >= 2

    pos_types = {info["type"] for info in results}
    assert "noun" in pos_types
    assert "verb" in pos_types

    verb_info = next(info for info in results if info["type"] == "verb")
    # Be robust to WordNet version changes
    assert any("operate" in d or "control" in d for d in verb_info["definitions"])
    assert any("sprint" in w or "move" in w for w in verb_info["related_words"])

    noun_info = next(info for info in results if info["type"] == "noun")
    assert any("trip" in d or "run" in d for d in noun_info["definitions"])


def test_get_word_info_from_wordnet_single_pos():
    """Word with a single POS (e.g., 'cat')."""
    results = dictionary_utils.get_word_info_from_wordnet("cat")
    assert len(results) >= 1
    noun_info = next(r for r in results if r["type"] == "noun")
    assert noun_info["type"] == "noun"
    assert any("feline" in h for h in noun_info["hypernyms_raw"])
    assert any("cat" in w for w in noun_info["related_words"])


# ---------------------------------------------------------------------
# get_pos_tag_simple
# ---------------------------------------------------------------------


def test_get_pos_tag_simple_nltk_success(monkeypatch):
    """Primary NLTK path works."""
    monkeypatch.setattr(
        dictionary_utils.nltk, "pos_tag", lambda words: [("test", "NN")]
    )
    assert dictionary_utils.get_pos_tag_simple("test") == "noun"


def test_get_pos_tag_simple_nltk_fails_spacy_success(monkeypatch):
    """Fallback to spaCy if NLTK fails."""
    monkeypatch.setattr(
        dictionary_utils.nltk, "pos_tag", MagicMock(side_effect=Exception("NLTK Error"))
    )

    mock_spacy_doc = MagicMock()
    mock_spacy_doc.__getitem__.return_value.pos_ = "VERB"

    mock_nlp = MagicMock(return_value=mock_spacy_doc)
    monkeypatch.setattr(dictionary_utils, "nlp", mock_nlp)

    assert dictionary_utils.get_pos_tag_simple("test") == "verb"


@pytest.mark.parametrize(
    ("spacy_pos", "expected_axiom_pos"),
    [
        ("PROPN", "noun"),
        ("NOUN", "noun"),
        ("VERB", "verb"),
        ("ADJ", "descriptor"),
        ("ADV", "adverb"),
    ],
)
def test_get_pos_tag_simple_spacy_mappings(
    monkeypatch, spacy_pos: str, expected_axiom_pos: str
):
    """Validate all spaCy POS to Axiom POS mappings."""
    monkeypatch.setattr(
        dictionary_utils.nltk, "pos_tag", MagicMock(side_effect=Exception)
    )

    mock_spacy_doc = MagicMock()
    mock_spacy_doc.__getitem__.return_value.pos_ = spacy_pos

    mock_nlp = MagicMock(return_value=mock_spacy_doc)
    monkeypatch.setattr(dictionary_utils, "nlp", mock_nlp)

    assert dictionary_utils.get_pos_tag_simple("test") == expected_axiom_pos


def test_get_pos_tag_simple_spacy_not_installed(monkeypatch):
    """If spaCy isn't installed (nlp=None), fallback to WordNet."""
    monkeypatch.setattr(
        dictionary_utils.nltk, "pos_tag", MagicMock(side_effect=Exception)
    )
    monkeypatch.setattr(dictionary_utils, "nlp", None)

    mock_synset = MagicMock()
    mock_synset.pos = lambda: "n"
    monkeypatch.setattr(wn, "synsets", lambda word: [mock_synset])

    assert dictionary_utils.get_pos_tag_simple("test") == "noun"


def test_get_pos_tag_simple_nltk_spacy_fail_wordnet_success(monkeypatch):
    """Test final fallback to WordNet."""
    monkeypatch.setattr(
        dictionary_utils.nltk, "pos_tag", MagicMock(side_effect=Exception)
    )
    monkeypatch.setattr(
        dictionary_utils, "nlp", MagicMock(__call__=MagicMock(side_effect=Exception))
    )

    mock_synset = MagicMock()
    mock_synset.pos = lambda: "v"
    monkeypatch.setattr(wn, "synsets", lambda word: [mock_synset])

    assert dictionary_utils.get_pos_tag_simple("test") == "verb"


# ---------------------------------------------------------------------
# lemmatize_word
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "pos", "expected"),
    [
        # No POS provided
        ("running", None, "run"),
        ("cats", None, "cat"),
        ("geese", None, "goose"),
        # With POS provided
        ("running", "verb", "run"),
        ("cats", "noun", "cat"),
        ("better", "descriptor", "good"),
        ("best", "adverb", "well"),
        # With POS that doesn't map
        ("corpora", "unknown_pos", "corpus"),
        # With POS that is a prefix
        ("nouns", "noun_phrase", "noun"),
    ],
)
def test_lemmatize_word(word: str, pos: str | None, expected: str):
    """Tests lemmatizer with and without POS tags."""
    assert dictionary_utils.lemmatize_word(word, pos) == expected
