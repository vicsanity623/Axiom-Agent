from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from axiom.knowledge_harvester import KnowledgeHarvester
from axiom.lexicon_manager import LexiconManager

if TYPE_CHECKING:
    from pathlib import Path

    import LogCaptureFixture
    import MonkeyPatch as PytestMonkeyPatch

    from axiom.cognitive_agent import CognitiveAgent


@pytest.fixture
def harvester(agent: CognitiveAgent) -> KnowledgeHarvester:
    """Provides a KnowledgeHarvester instance for tests using the global agent fixture."""
    lock = threading.Lock()
    return KnowledgeHarvester(agent, lock)


def test_load_and_save_research_cache(tmp_path: Path, harvester: KnowledgeHarvester):
    """
    Given: A harvester and a temporary path for its cache.
    When: A term is marked as researched.
    Then: The cache file is created with the correct content and can be reloaded.
    """
    # Arrange
    cache_file = tmp_path / "test_cache.json"
    harvester.cache_path = cache_file
    term_to_research = "epistemology"

    # Act
    harvester._mark_as_researched(term_to_research)

    # Assert: File was saved correctly
    assert cache_file.exists()
    with cache_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert term_to_research in data

    # Act: Reload the cache
    harvester.researched_terms = set()  # Reset before loading
    harvester._load_research_cache()

    # Assert: Cache was loaded correctly
    assert term_to_research in harvester.researched_terms


def test_load_research_cache_errors(
    tmp_path: Path, harvester: KnowledgeHarvester, caplog: LogCaptureFixture
):
    """
    Given: A harvester and a malformed or non-existent cache file.
    When: The cache is loaded.
    Then: Errors are handled gracefully and logged appropriately.
    """
    # Scenario 1: File not found (should be silent)
    harvester.cache_path = tmp_path / "non_existent.json"
    harvester._load_research_cache()
    assert "Failed to load research cache" not in caplog.text

    # Scenario 2: Invalid JSON content
    cache_file = tmp_path / "invalid.json"
    cache_file.write_text("this is not json")
    harvester.cache_path = cache_file
    harvester._load_research_cache()
    assert "Failed to load research cache" in caplog.text
    assert harvester.researched_terms == set()

    # Scenario 3: Malformed JSON (wrong data type)
    cache_file.write_text('{"key": "value"}')  # Should be a list
    harvester.researched_terms = {"initial"}
    harvester._load_research_cache()
    assert "Cache file is malformed" in caplog.text
    assert harvester.researched_terms == {"initial"}  # Should not be cleared


def test_prune_research_cache(
    harvester: KnowledgeHarvester, monkeypatch: PytestMonkeyPatch
):
    """
    Given: A research cache with known and unknown terms.
    When: _prune_research_cache is called.
    Then: Only terms that are now known by the lexicon are removed.
    """
    # Arrange
    harvester.researched_terms = {"known_term", "unknown_term", "another_known"}
    monkeypatch.setattr(
        LexiconManager,
        "is_known_word",
        lambda self, term: term in ["known_term", "another_known"],
    )
    mock_save = MagicMock()
    monkeypatch.setattr(KnowledgeHarvester, "_save_research_cache", mock_save)

    # Act
    harvester._prune_research_cache()

    # Assert
    assert harvester.researched_terms == {"unknown_term"}
    assert mock_save.call_count == 1


def test_discover_cycle(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch: PytestMonkeyPatch
):
    """
    Given: A harvester with a mocked topic finder.
    When: The discover cycle is run.
    Then: An "INVESTIGATE" goal for the new topic is added to the agent.
    """
    # Arrange
    monkeypatch.setattr(
        KnowledgeHarvester, "_find_new_topic", lambda self, max_attempts=5: "ontology"
    )
    agent.learning_goals.clear()

    # Act
    harvester.discover_cycle()

    # Assert
    assert "INVESTIGATE: ontology" in agent.learning_goals
