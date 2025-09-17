# dictionary_utils.py

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Mapping NLTK POS tags to our ConceptNode types
pos_map = {
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
    "RBS": "adverb",
    "SYM": "symbol",
    "TO": "to",
    "UH": "interjection",
    "CD": "number",
    "POS": "possessive",
}


def get_word_info_from_wordnet(word):
    """
    Retrieves basic information (POS, definitions, hypernyms) for a word from WordNet.
    Returns a dict {type, definitions, hypernyms_raw, related_words}.
    """
    word_info = {
        "type": "concept",
        "definitions": [],
        "hypernyms_raw": [],
        "related_words": [],
    }

    synsets = wn.synsets(word.lower())

    if not synsets:
        return word_info

    best_synset = None
    for ss in synsets:
        if ss.pos() == "n":
            best_synset = ss
            break
        elif ss.pos() == "v" and not best_synset:
            best_synset = ss
        elif (
            ss.pos() == "a" or ss.pos() == "s" and not best_synset
        ):  # 's' is adjective satellite
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


def get_pos_tag_simple(word):
    """
    A simpler POS tagger. Tries NLTK's pos_tag, but falls back to WordNet's primary POS
    or a generic 'concept' if the tagger resource is missing.
    """
    try:
        tagged_word = nltk.pos_tag([word])
        if tagged_word:
            return pos_map.get(tagged_word[0][1], "concept")
    except LookupError:
        print(
            f"WARNING: NLTK pos_tagger resource missing for '{word}'. Falling back to WordNet primary POS."
        )
        # Fallback: Use WordNet's primary POS for the word if pos_tagger fails
        synsets = wn.synsets(word.lower())
        if synsets:
            best_synset = None
            for ss in synsets:
                if ss.pos() == "n":
                    best_synset = ss
                    break
                elif ss.pos() == "v" and not best_synset:
                    best_synset = ss
                elif ss.pos() == "a" or ss.pos() == "s" and not best_synset:
                    best_synset = ss
            if not best_synset and synsets:
                best_synset = synsets[0]

            if best_synset:
                nltk_pos = best_synset.pos()
                if nltk_pos == "n":
                    return "noun"
                elif nltk_pos == "v":
                    return "verb"
                elif nltk_pos == "a" or nltk_pos == "s":
                    return "descriptor"
                elif nltk_pos == "r":
                    return "adverb"
        return "concept"  # Final fallback if WordNet also doesn't give a specific POS
    except Exception as e:
        print(
            f"An unexpected error occurred in get_pos_tag_simple for '{word}': {e}. Falling back to 'concept'."
        )
        return "concept"  # General error fallback
    return "concept"  # Default if no specific POS found


def lemmatize_word(word, pos=None):
    """Lemmatizes a word."""
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
