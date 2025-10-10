from __future__ import annotations

# dictionary_utils.py
from typing import Final, Literal, TypedDict

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

pos_map: Final = {
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

try:
    wn.synsets("test")
except LookupError:
    print("NLTK 'wordnet' corpus not found. Downloading now...")
    nltk.download('wordnet')
    from nltk.corpus import wordnet as wn

class WordInfo(TypedDict):
    type: Literal["concept" | "noun" | "verb" | "descriptor" | "adverb"]
    definitions: list[str]
    hypernyms_raw: list[str]
    related_words: list[str]


def get_word_info_from_wordnet(word: str) -> WordInfo:
    """Retrieve detailed linguistic information for a word from WordNet.

    This function queries WordNet for a given word and attempts to find
    its most likely part of speech, definitions, and hypernyms (more
    general concepts, e.g., 'dog' -> 'canine'). It also extracts related
    words from the definitions and lemmas.

    It prioritizes nouns, then verbs, then adjectives when selecting the
    primary "synset" (sense of the word) to analyze.

    Args:
        word: The single word to look up.

    Returns:
        A `WordInfo` TypedDict containing the extracted type, definitions,
        hypernyms, and related words. Returns a default 'concept' type
        if the word is not found.
    """
    word_info = WordInfo(
        {
            "type": "concept",
            "definitions": [],
            "hypernyms_raw": [],
            "related_words": [],
        },
    )

    synsets = wn.synsets(word.lower())

    if not synsets:
        return word_info

    best_synset = None
    for ss in synsets:
        if ss.pos() == "n":
            best_synset = ss
            break
        if ss.pos() == "v" and not best_synset:
            best_synset = ss
        elif ss.pos() == "a" or ss.pos() == "s" and not best_synset:
            best_synset = ss

    if not best_synset and synsets:
        best_synset = synsets[0]

    if best_synset:
        nltk_pos = best_synset.pos()
        if nltk_pos == "n":
            word_info["type"] = "noun"
        elif nltk_pos == "v":
            word_info["type"] = "verb"
        elif nltk_pos == "a" or nltk_pos == "s":
            word_info["type"] = "descriptor"
        elif nltk_pos == "r":
            word_info["type"] = "adverb"

        word_info["definitions"].append(best_synset.definition())
        for example in best_synset.examples():
            if example not in word_info["definitions"]:
                word_info["definitions"].append(example)

        for hypernym_synset in best_synset.hypernyms():
            for lemma in hypernym_synset.lemmas():
                if lemma.name().replace("_", " ") not in word_info["hypernyms_raw"]:
                    word_info["hypernyms_raw"].append(lemma.name().replace("_", " "))

        for definition in word_info["definitions"]:
            tokens = definition.lower().split()
            for token_raw in tokens:
                token = token_raw.strip(".,;?!\"'()[]{}")
                if (
                    token.isalpha()
                    and len(token) > 2
                    and token != word.lower()
                    and token not in word_info["related_words"]
                ):
                    word_info["related_words"].append(token)

        for lemma in best_synset.lemmas():
            if (
                lemma.name().replace("_", " ") != word.lower()
                and lemma.name().replace("_", " ") not in word_info["related_words"]
            ):
                word_info["related_words"].append(lemma.name().replace("_", " "))

    return word_info


def get_pos_tag_simple(word: str) -> str:
    """Determine the part of speech for a word using a fallback strategy.

    This function first attempts to use NLTK's fast `pos_tag` function.
    If the required NLTK resource is not downloaded, it gracefully falls
    back to querying WordNet for the word's primary part of speech.

    If both methods fail, it returns the generic type 'concept'.

    Args:
        word: The single word to tag.

    Returns:
        A string representing the determined part of speech (e.g., 'noun',
        'verb', 'concept').
    """
    try:
        tagged_word = nltk.pos_tag([word])
        if tagged_word:
            return pos_map.get(tagged_word[0][1], "concept")
    except LookupError:
        print(
            f"WARNING: NLTK pos_tagger resource missing for '{word}'. Falling back to WordNet primary POS.",
        )
        synsets = wn.synsets(word.lower())
        if synsets:
            best_synset = None
            for ss in synsets:
                if ss.pos() == "n":
                    best_synset = ss
                    break
                if ss.pos() == "v" and not best_synset:
                    best_synset = ss
                elif ss.pos() == "a" or ss.pos() == "s" and not best_synset:
                    best_synset = ss
            if not best_synset and synsets:
                best_synset = synsets[0]

            if best_synset:
                nltk_pos = best_synset.pos()
                if nltk_pos == "n":
                    return "noun"
                if nltk_pos == "v":
                    return "verb"
                if nltk_pos == "a" or nltk_pos == "s":
                    return "descriptor"
                if nltk_pos == "r":
                    return "adverb"
    except Exception as e:
        print(
            f"An unexpected error occurred in get_pos_tag_simple for '{word}': {e}. Falling back to 'concept'.",
        )
    return "concept"


def lemmatize_word(word: str, pos: str | None = None) -> WordNetLemmatizer:
    """Reduce a word to its base or dictionary form (lemma).

    Uses the WordNetLemmatizer to convert a word to its root form.
    For example, 'running' becomes 'run', and 'cats' becomes 'cat'.
    Providing the part of speech (POS) can improve accuracy.

    Args:
        word: The word to lemmatize.
        pos: An optional part-of-speech tag (e.g., 'n', 'v', 'a', 'r').

    Returns:
        The lemmatized form of the word as a string.
    """
    if pos:
        wn_pos = None
        if pos.startswith("n"):
            wn_pos = wn.NOUN
        elif pos.startswith("v"):
            wn_pos = wn.VERB
        elif pos.startswith("a"):
            wn_pos = wn.ADJ
        elif pos.startswith("r"):
            wn_pos = wn.ADV
        if wn_pos:
            return lemmatizer.lemmatize(word, wn_pos)
    return lemmatizer.lemmatize(word)
