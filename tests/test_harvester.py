import threading
from unittest.mock import MagicMock

import pytest
import requests
import wikipedia

from axiom.cognitive_agent import CognitiveAgent
from axiom.graph_core import ConceptNode
from axiom.knowledge_harvester import KnowledgeHarvester

# --- Fixtures needed ONLY by the tests in this file ---


@pytest.fixture
def harvester(agent: CognitiveAgent) -> KnowledgeHarvester:
    """Creates a KnowledgeHarvester instance linked to the test agent."""
    lock = threading.Lock()
    return KnowledgeHarvester(agent, lock)


# --- All the KnowledgeHarvester Tests ---


class MockWikipediaPage:
    def __init__(self, title, summary):
        self.title = title
        self.summary = summary


def test_discover_cycle_adds_goal(harvester, agent, monkeypatch):
    # Use monkeypatch to patch read-only method
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester._find_new_topic",
        lambda self: "Quantum Mechanics",
    )

    agent.learning_goals.clear()
    harvester.discover_cycle()
    assert any("Quantum Mechanics" in goal for goal in agent.learning_goals)
    print("KnowledgeHarvester: discover_cycle added a new topic successfully.")


def test_resolve_investigation_goal_with_api_success(harvester, agent, monkeypatch):
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_definition_from_api",
        lambda self, word: ("noun", "A test definition."),
    )

    goal = "INVESTIGATE: testword"
    agent.learning_goals.append(goal)

    resolved = harvester._resolve_investigation_goal(goal)
    assert resolved is True
    assert goal not in agent.learning_goals
    print("KnowledgeHarvester: _resolve_investigation_goal succeeded via API.")


def test_resolve_investigation_goal_fallback_web(
    harvester: KnowledgeHarvester,
    agent: CognitiveAgent,
    monkeypatch,
):
    """
    Tests the web fallback in _resolve_investigation_goal, mocking the
    final learning step to isolate the harvester's logic.
    """
    # 1. GIVEN: All external APIs will fail except for Wikipedia.
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_definition_from_api",
        lambda self, word: None,
    )
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_wikipedia",
        lambda self, topic: ("A Test Topic", "The test topic is a noun."),
    )
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_duckduckgo",
        lambda self, topic: None,
    )

    # AND: We mock the agent's learning function to always succeed.
    # This is the key to testing the harvester's logic in isolation.
    learn_spy = MagicMock(return_value=True)
    monkeypatch.setattr(agent, "learn_new_fact_autonomously", learn_spy)

    goal = "INVESTIGATE: fallbackword"
    agent.learning_goals.append(goal)

    # 2. WHEN: We call the method under test.
    resolved = harvester._resolve_investigation_goal(goal)

    # 3. THEN: Assert that the goal was resolved.
    assert resolved is True
    assert goal not in agent.learning_goals

    # AND: Assert that the agent's learning function was called with the correct data.
    learn_spy.assert_called_once_with(
        fact_sentence="The test topic is a noun.",
        source_topic="A Test Topic",
    )

    print(
        "✅ Harvester correctly fell back to web search and called the learning function.",
    )


def test_study_cycle_resolves_goal(harvester, agent, monkeypatch):
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester._resolve_investigation_goal",
        lambda self, goal: False,
    )
    agent.learning_goals.append("INVESTIGATE: someword")
    harvester.study_cycle()
    assert len(agent.learning_goals) == 0
    print("KnowledgeHarvester: study_cycle resolved an investigation goal.")


def test_refinement_cycle_no_chunky_fact(harvester, agent, monkeypatch):
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester._find_chunky_fact",
        lambda self: None,
    )

    # Should not raise exceptions
    harvester.refinement_cycle()
    print("KnowledgeHarvester: refinement_cycle handled no chunky facts gracefully.")


@pytest.mark.parametrize(
    "graph_nodes",
    [
        [("apple", "fruit"), ("fruit", "category")],
        [("socrates", "philosopher"), ("philosopher", "person")],
    ],
)
def test_deepen_knowledge_of_random_concept_adds_goal(
    harvester,
    agent,
    graph_nodes,
    monkeypatch,
):
    agent.graph.graph.clear()
    agent.learning_goals.clear()
    for name, node_type in graph_nodes:
        agent.graph.add_node(ConceptNode(name=name, node_type=node_type))

    # Pre-promote the words that will be in our mocked fact.
    agent.lexicon._promote_word_for_test("apple", "noun")
    agent.lexicon._promote_word_for_test("socrates", "noun")
    agent.lexicon._promote_word_for_test("test", "noun")  # It's a noun in this context
    agent.lexicon._promote_word_for_test("concept", "noun")

    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_wikipedia",
        lambda self, topic: (topic, f"{topic} is a test concept."),
    )
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_duckduckgo",
        lambda self, topic: None,
    )
    monkeypatch.setattr("axiom.knowledge_harvester.random.choice", lambda x: x[0])

    harvester._deepen_knowledge_of_random_concept()

    test_node = agent.graph.get_node_by_name("test concept")
    assert test_node is not None, (
        "The agent failed to learn the fact from the web search."
    )


def test_find_chunky_fact_returns_none_when_no_edges(harvester, agent):
    """
    Ensures _find_chunky_fact returns None if there are no edges in the graph.
    """
    # Clear all edges
    for edge in list(agent.graph.graph.edges):
        agent.graph.graph.remove_edge(*edge)
    result = harvester._find_chunky_fact()
    assert result is None
    print("KnowledgeHarvester: _find_chunky_fact returned None for empty graph.")


def test_find_chunky_fact_returns_edge_with_highest_weight(
    harvester,
    agent,
    monkeypatch,
):
    """
    Tests that _find_chunky_fact finds a suitable "chunky" fact,
    which is an 'is_a' edge with a long target name.
    """
    agent.graph.graph.clear()
    node_a = agent.graph.add_node(ConceptNode(name="A"))

    # KEY: The name must have MORE THAN 5 words to be "chunky".
    # We will use a 6-word phrase.
    long_name_node = agent.graph.add_node(
        ConceptNode(name="a very very long definition phrase"),
    )

    # Add a non-chunky edge (should be ignored)
    agent.graph.add_edge(node_a, node_a, "related_to", weight=0.9)
    # Add the chunky edge we expect to find
    chunky_edge_to_find = agent.graph.add_edge(
        node_a,
        long_name_node,
        "is_a",
        weight=0.9,
    )

    # Mock random.choice to make the test deterministic
    monkeypatch.setattr("axiom.knowledge_harvester.random.choice", lambda x: x[0])

    chunky_result = harvester._find_chunky_fact()

    # 1. Verify that a fact was found
    assert chunky_result is not None

    # 2. Verify that the correct fact was found
    source_node, target_node, edge = chunky_result
    assert source_node.name == "a"
    assert (
        target_node.name == "a very very long definition phrase"
    )  # <-- UPDATED ASSERTION
    assert edge.id == chunky_edge_to_find.id  # <-- MORE ROBUST CHECK
    assert edge.type == "is_a"

    print(
        "KnowledgeHarvester: _find_chunky_fact selected the chunky edge successfully.",
    )


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.RequestException("Mocked HTTP Error")


def test_harvester_gets_fact_from_wikipedia(
    harvester: KnowledgeHarvester,
    agent: CognitiveAgent,
    monkeypatch,
):
    """
    Tests the get_fact_from_wikipedia method, mocking the interpreter's
    verification step to isolate the harvester's logic.
    """
    # 1. GIVEN: We mock the external dependencies.
    # Mock the wikipedia library to return a predictable page.
    mock_page = MockWikipediaPage(
        title="Python (programming language)",
        summary="Python is a programming language.",
    )
    monkeypatch.setattr(wikipedia, "search", lambda *args, **kwargs: [mock_page.title])
    monkeypatch.setattr(wikipedia, "page", lambda *args, **kwargs: mock_page)

    # AND: We mock the interpreter's verification method to return a predictable, reframed fact.
    reframed_fact = "Python is a language."
    verify_spy = MagicMock(return_value=reframed_fact)
    monkeypatch.setattr(agent.interpreter, "verify_and_reframe_fact", verify_spy)

    # 2. WHEN: We call the method under test.
    result = harvester.get_fact_from_wikipedia("Python")

    # 3. THEN: Assert that the method returned the reframed fact.
    assert result is not None
    title, fact = result
    assert title == mock_page.title
    assert fact == reframed_fact

    # AND: Assert that our verification spy was called with the correct raw sentence.
    verify_spy.assert_called_once_with(
        original_topic="Python",
        raw_sentence="Python is a programming language.",
    )
    print("✅ Wikipedia harvester correctly used the LLM verifier.")


@pytest.mark.parametrize(
    ("api_response", "should_succeed"),
    [
        # "Happy path": API returns a good definition
        (
            {"AbstractText": "A programming language..."},
            True,
        ),
        # "Sad path": API returns an empty response
        ({}, False),
    ],
)
def test_harvester_gets_fact_from_duckduckgo(
    harvester: KnowledgeHarvester,
    agent: CognitiveAgent,
    monkeypatch,
    api_response,
    should_succeed,
):
    """
    Tests the get_fact_from_duckduckgo method, mocking the interpreter's
    verification step.
    """
    # 1. GIVEN: Mock the external dependencies.
    # Mock the requests library to return a predictable API response.
    mock_api_response = {"AbstractText": "Python is a programming language."}
    mock_response_obj = MockResponse(mock_api_response)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response_obj)

    # AND: We mock the interpreter's verification method.
    reframed_fact = "Python is a language."
    verify_spy = MagicMock(return_value=reframed_fact)
    monkeypatch.setattr(agent.interpreter, "verify_and_reframe_fact", verify_spy)

    # 2. WHEN: We call the method under test.
    result = harvester.get_fact_from_duckduckgo("Python")

    # 3. THEN: Assert that the method returned the reframed fact.
    assert result is not None
    topic, fact = result
    assert topic == "Python"
    assert fact == reframed_fact

    # AND: Assert that our verification spy was called correctly.
    verify_spy.assert_called_once_with(
        original_topic="Python",
        raw_sentence="Python is a programming language.",
    )
    print("✅ DuckDuckGo harvester correctly used the LLM verifier.")


def test_harvester_handles_complete_failure(harvester, agent, monkeypatch):
    """
    Covers the failure paths in the KnowledgeHarvester where all external
    APIs and web searches fail to produce a result.
    """
    topic = "nonexistenttopic"
    goal = f"INVESTIGATE: {topic}"
    agent.learning_goals.append(goal)

    # 1. Test _resolve_investigation_goal failure
    # Mock all external sources to return None
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_definition_from_api",
        lambda self, word: None,
    )
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_wikipedia",
        lambda self, topic: None,
    )
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_duckduckgo",
        lambda self, topic: None,
    )

    # Action & Verification
    resolved = harvester._resolve_investigation_goal(goal)
    assert resolved is False, "Goal should not be resolved when all sources fail."
    # The goal should still be in the list, as it wasn't resolved
    assert goal in agent.learning_goals
    print("Harvester correctly handled failure of all knowledge sources.")

    # 2. Test discover_cycle failure
    # Mock _find_new_topic to return None
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester._find_new_topic",
        lambda self: None,
    )

    initial_goal_count = len(agent.learning_goals)
    harvester.discover_cycle()

    # Verification: No new goals should be added
    assert len(agent.learning_goals) == initial_goal_count
    print("Harvester discover_cycle correctly handled finding no new topic.")


@pytest.mark.parametrize(
    ("raw_sentence", "expected_clean_sentence"),
    [
        # Test case 1: Introductory phrase
        (
            "In psychology, confusion is a state of being unclear.",
            "confusion is a state of being unclear.",
        ),
        # Test case 2: Parenthetical text and leading junk
        (
            ": supernovae) is a powerful and luminous stellar explosion.",
            "is a powerful and luminous stellar explosion.",
        ),
        # Test case 3: Semicolon splitting
        (
            "Comprise the kingdom plantae; landmass is a region.",
            "Comprise the kingdom plantae",
        ),
        # Test case 4: A clean sentence that should be mostly unchanged
        ("The sun is a star.", "The sun is a star."),
        # Test case 5: Leading hyphen
        ("- A fact with a leading hyphen.", "A fact with a leading hyphen."),
    ],
)
def test_agent_sanitizes_sentences_correctly(  # <-- Renamed function for clarity
    agent: CognitiveAgent,  # <-- Use the 'agent' fixture now
    raw_sentence: str,
    expected_clean_sentence: str,
):
    """
    Covers the _sanitize_sentence_for_learning helper method, now in CognitiveAgent.
    """
    # 1. WHEN: We call the sanitizer method on the AGENT object.
    clean_sentence = agent._sanitize_sentence_for_learning(raw_sentence)

    # 2. THEN: Assert that the output matches the expected clean version.
    assert clean_sentence == expected_clean_sentence
