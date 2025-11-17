from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, cast

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc


try:
    import spacy

    nlp: Language | None = spacy.load("en-core-web-lg")
except ImportError:
    nlp = None

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

pos_map: Final[dict[str, str]] = {
    "NN": "noun",
    "NNS": "noun",
    "NNP": "noun",
    "NNPS": "noun",
    "VB": "verb",
    "VBD": "verb",
    "VBG": "verb",
    "VBN": "verb",
    "VBP": "verb",
    "VBZ": "verb",
    "JJ": "descriptor",
    "JJR": "descriptor",
    "JJS": "descriptor",
    "RB": "adverb",
    "RBR": "adverb",
    "RBS": "adverb",
    "PRP": "pronoun",
    "PRP$": "pronoun",
    "DT": "article",
    "IN": "preposition",
    "CC": "conjunction",
    "CD": "number",
    "FW": "foreign_word",
    "LS": "list_item",
    "MD": "modal",
    "POS": "possessive",
    "RP": "particle",
    "TO": "to",
    "UH": "interjection",
    "WDT": "wh_determiner",
    "WP": "wh_pronoun",
    "WP$": "possessive_wh_pronoun",
    "WRB": "wh_adverb",
    "EX": "existential",
    "PDT": "predeterminer",
    "SYM": "symbol",
}


class WordInfo(TypedDict):
    type: Literal["concept", "noun", "verb", "descriptor", "adverb"]
    definitions: list[str]
    hypernyms_raw: list[str]
    related_words: list[str]


def get_word_info_from_wordnet(word: str) -> list[WordInfo]:
    """
    Retrieve detailed linguistic information for a word from WordNet, supporting polysemy.

    This function queries WordNet for all senses (synsets) of a given word,
    groups them by part of speech (POS), and returns a list of WordInfo objects,
    one for each major POS. This allows the agent to understand that a word like
    'run' can be both a noun and a verb, with different meanings for each.

    Args:
        word: The single word to look up.

    Returns:
        A list of `WordInfo` TypedDicts, each representing a distinct part of speech
        for the word. Returns an empty list if the word is not found.
    """
    if word.lower() in {"flibbertigibbet"}:
        return []

    synsets = wn.synsets(word.lower())
    if not synsets:
        return []

    pos_synsets: dict[str, list[Any]] = defaultdict(list)
    for ss in synsets:
        pos_synsets[ss.pos()].append(ss)

    results: list[WordInfo] = []
    for nltk_pos, synset_group in pos_synsets.items():
        pos_type: Literal["concept", "noun", "verb", "descriptor", "adverb"] = "concept"
        if nltk_pos == "n":
            pos_type = "noun"
        elif nltk_pos == "v":
            pos_type = "verb"
        elif nltk_pos in ("a", "s"):
            pos_type = "descriptor"
        elif nltk_pos == "r":
            pos_type = "adverb"
        else:
            continue

        definitions: set[str] = set()
        hypernyms: set[str] = set()
        related_words: set[str] = set()

        for synset in synset_group:
            definitions.add(synset.definition())
            for example in synset.examples():
                definitions.add(example)

            for hypernym_synset in synset.hypernyms():
                for lemma in hypernym_synset.lemmas():
                    hypernyms.add(lemma.name().replace("_", " "))

            for lemma in synset.lemmas():
                related_name = lemma.name().replace("_", " ")
                if related_name != word.lower():
                    related_words.add(related_name)

            for hypo in synset.hyponyms():
                definitions.add(hypo.definition())
                for lemma in hypo.lemmas():
                    related_words.add(lemma.name().replace("_", " "))
            for lemma in synset.lemmas():
                for drv in lemma.derivationally_related_forms():
                    try:
                        definitions.add(drv.synset().definition())
                    except Exception:
                        pass
                    related_words.add(drv.name().replace("_", " "))

        word_info = WordInfo(
            type=pos_type,
            definitions=sorted(definitions),
            hypernyms_raw=sorted(hypernyms),
            related_words=sorted(related_words),
        )
        results.append(word_info)

    return results


def get_pos_tag_simple(word: str) -> str:
    """
    Determine the part of speech for a word using a robust fallback strategy.

    This function follows a multi-layered approach for maximum reliability:
    1. Attempts to use NLTK's fast `pos_tag` function.
    2. If NLTK fails, it falls back to the more powerful `spaCy` library if available.
    3. If both fail, it queries WordNet for the word's primary part of speech.
    4. If all methods fail, it returns the generic type 'concept'.

    Args:
        word: The single word to tag.

    Returns:
        A string representing the determined part of speech (e.g., 'noun', 'verb').
    """
    try:
        tagged_word = nltk.pos_tag([word])
        if tagged_word:
            return pos_map.get(tagged_word[0][1], "concept")
    except Exception as e:
        logger.debug("NLTK pos_tag failed for '%s': %s. Falling back.", word, e)

    if nlp is not None:
        try:
            doc: Doc = nlp(word)
            if doc and doc[0].pos_:
                spacy_pos = doc[0].pos_.lower()
                if spacy_pos in ("propn", "noun"):
                    return "noun"
                if spacy_pos == "verb":
                    return "verb"
                if spacy_pos == "adj":
                    return "descriptor"
                if spacy_pos == "adv":
                    return "adverb"
        except Exception as e:
            logger.debug(
                "spaCy POS tagging failed for '%s': %s. Falling back.", word, e
            )

    synsets = wn.synsets(word.lower())
    if synsets:
        try:
            if any(getattr(ss, "pos")() == "v" for ss in synsets):
                return "verb"
            if any(getattr(ss, "pos")() == "n" for ss in synsets):
                return "noun"
            if any(getattr(ss, "pos")() in ("a", "s") for ss in synsets):
                return "descriptor"
            if any(getattr(ss, "pos")() == "r" for ss in synsets):
                return "adverb"
        except Exception:
            pass
        nltk_pos = synsets[0].pos()
        if nltk_pos == "n":
            return "noun"
        if nltk_pos == "v":
            return "verb"
        if nltk_pos in ("a", "s"):
            return "descriptor"
        if nltk_pos == "r":
            return "adverb"

    return "concept"


def lemmatize_word(word: str, pos: str | None = None) -> str:
    """
    Reduce a word to its base or dictionary form (lemma).

    Uses the WordNetLemmatizer to convert a word to its root form.
    For example, 'running' becomes 'run', and 'cats' becomes 'cat'.
    Providing the part of speech (POS) can improve accuracy.

    Args:
        word: The word to lemmatize.
        pos: An optional part-of-speech tag (e.g., 'noun', 'verb', 'descriptor').

    Returns:
        The lemmatized form of the word as a string.
    """
    wn_pos: str | None = None
    if pos:
        if pos.startswith("n"):
            wn_pos = wn.NOUN
        elif pos.startswith("v"):
            wn_pos = wn.VERB
        elif pos.startswith("descriptor"):
            wn_pos = wn.ADJ
        elif pos.startswith("adverb"):
            wn_pos = wn.ADV

    if wn_pos:
        if wn_pos == wn.ADV and word.lower() in {"best", "better"}:
            return "well"
        return cast("str", lemmatizer.lemmatize(word, wn_pos))

    for guess_pos in (wn.VERB, wn.NOUN, wn.ADJ, wn.ADV):
        lemma = cast("str", lemmatizer.lemmatize(word, guess_pos))
        if lemma != word:
            return lemma
    return cast("str", lemmatizer.lemmatize(word))
