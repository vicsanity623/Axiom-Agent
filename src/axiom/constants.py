from typing import Final

RELATION_TYPE_MAP: Final[dict[str, str]] = {
    "be": "is_a",
    "is": "is_a",
    "are": "is_a",
    "cause": "causes",
    "causes": "causes",
    "locate_in": "is_located_in",
    "located_in": "is_located_in",
    "part_of": "is_part_of",
    "learn": "learns",
    "release": "released",
    "released": "released",
}

CONTRACTION_MAP: Final[dict[str, str]] = {
    "what's": "what is",
    "whats": "what is",
    "it's": "it is",
    "its": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
}

PROVENANCE_RANK: Final[dict[str, int]] = {
    "seed": 5,
    "dictionary_api": 4,
    "llm_verified": 3,
    "wikipedia": 2,
    "duckduckgo": 2,
    "user_input": 1,
    "llm": 1,
}
