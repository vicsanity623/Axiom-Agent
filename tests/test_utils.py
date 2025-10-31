from __future__ import annotations

import importlib
from unittest.mock import MagicMock

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
