from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from axiom.knowledge_harvester import KnowledgeHarvester

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent


@pytest.fixture
def harvester(agent: CognitiveAgent) -> KnowledgeHarvester:
    """Provides a KnowledgeHarvester instance for tests."""
    lock = threading.Lock()
    return KnowledgeHarvester(agent, lock)


def test_save_research_cache_os_error(
    harvester: KnowledgeHarvester, monkeypatch: Any, caplog: Any
) -> None:
    """
    Given: A harvester.
    When: Saving the research cache raises an OSError.
    Then: An error is logged.
    """
    # Covers lines 96-97
    monkeypatch.setattr(Path, "open", MagicMock(side_effect=OSError("Disk full")))
    harvester._save_research_cache()
    assert "Failed to save research cache" in caplog.text


@patch("axiom.knowledge_harvester.KnowledgeHarvester.get_definition_from_api")
def test_resolve_investigation_goal_single_word_api_success(
    mock_api: MagicMock,
    agent: CognitiveAgent,
    harvester: KnowledgeHarvester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given: A single-word investigation goal.
    When: The dictionary API returns a valid definition and POS.
    Then: The agent learns the facts and the goal is resolved.
    """
    # Covers lines 162-205
    goal = "INVESTIGATE: testword"
    mock_api.return_value = ("noun", "a word for testing")

    # FIX: Use monkeypatch.setattr for type-safe mocking instead of direct assignment.
    mock_learn = MagicMock(return_value=True)
    monkeypatch.setattr(agent, "learn_new_fact_autonomously", mock_learn)

    agent.learning_goals.append(goal)

    result = harvester._resolve_investigation_goal(goal)

    assert result is True
    assert goal not in agent.learning_goals
    # Verify that the learning function was actually called.
    mock_learn.assert_called()


def test_prune_research_cache_no_terms(harvester: KnowledgeHarvester) -> None:
    """
    Given: A harvester with an empty set of researched terms.
    When: _prune_research_cache is called.
    Then: The function returns early without error.
    """
    # Covers lines 436-437
    harvester.researched_terms.clear()
    # This test just needs to run without raising an exception
    harvester._prune_research_cache()
    assert not harvester.researched_terms


@patch("axiom.knowledge_harvester.wikipedia.search")
def test_find_new_topic_handles_empty_search_results(
    mock_search: MagicMock, harvester: KnowledgeHarvester
) -> None:
    """
    Given: A harvester where wikipedia search returns no results.
    When: _find_new_topic is called.
    Then: It continues to the next attempt without error.
    """
    # Covers lines 637-638
    mock_search.return_value = []
    # We expect it to return None after all attempts fail
    assert harvester._find_new_topic(max_attempts=1) is None


def test_get_definition_from_api_http_error(
    harvester: KnowledgeHarvester, monkeypatch: Any
) -> None:
    """
    Given: A harvester where the dictionary API returns a non-200 status.
    When: get_definition_from_api is called.
    Then: It returns None.
    """
    # Covers lines 730-731
    mock_response = MagicMock()
    mock_response.status_code = 404
    monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)
    assert harvester.get_definition_from_api("word") is None


def test_get_fact_from_duckduckgo_with_definition(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch: Any
) -> None:
    """
    Given: A harvester where the DuckDuckGo API returns a valid definition.
    When: get_fact_from_duckduckgo is called.
    Then: A reframed fact is returned.
    """
    # Covers lines 816-831
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"Definition": "A testable assertion."}
    monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)
    monkeypatch.setattr(
        agent.interpreter,
        "verify_and_reframe_fact",
        lambda original_topic, raw_sentence: raw_sentence,
    )

    result = harvester.get_fact_from_duckduckgo("test")
    assert result is not None
    assert result[1] == "A testable assertion."
