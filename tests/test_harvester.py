import threading

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


@pytest.mark.parametrize(
    ("summary_text", "should_succeed"),
    [
        (
            "Python is an interpreted, high-level and general-purpose programming language. It was created by...",
            True,
        ),
        (
            "This is an exceptionally long and convoluted sentence that contains far too many words, clauses, and commas, which is designed specifically to fail the simplicity check that is built into the knowledge harvester.",
            False,
        ),
    ],
)
def test_harvester_gets_fact_from_wikipedia(
    harvester,
    monkeypatch,
    summary_text,
    should_succeed,
):
    topic = "Python"
    mock_page = MockWikipediaPage(
        title="Python (programming language)",
        summary=summary_text,
    )
    monkeypatch.setattr(wikipedia, "search", lambda query, results: [mock_page.title])
    monkeypatch.setattr(
        wikipedia,
        "page",
        lambda title, auto_suggest, redirect: mock_page,
    )
    result = harvester.get_fact_from_wikipedia(topic)
    if should_succeed:
        assert result is not None
        title, fact = result
        assert title == mock_page.title
        assert (
            fact
            == "Python is an interpreted, high-level and general-purpose programming language."
        )
    else:
        assert result is None


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
    # CORRECT: Patch the class, not the instance
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
    harvester,
    agent,
    monkeypatch,
):  # Add monkeypatch
    # CORRECT: Patch the class to simulate API failure
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_definition_from_api",
        lambda self, word: None,
    )

    # Patch Wikipedia and DuckDuckGo (these are fine as they are not __slots__)
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_wikipedia",
        lambda self, topic: (topic, f"{topic} is a test noun."),
    )
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_duckduckgo",
        lambda self, topic: None,
    )

    goal = "INVESTIGATE: fallbackword"
    agent.learning_goals.append(goal)

    resolved = harvester._resolve_investigation_goal(goal)
    assert resolved is True
    assert goal not in agent.learning_goals
    print(
        "KnowledgeHarvester: _resolve_investigation_goal succeeded via fallback web search.",
    )


def test_study_cycle_resolves_goal(harvester, agent, monkeypatch):
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester._resolve_investigation_goal",
        lambda self, goal: False,  # Mock failure to trigger removal logic
    )
    agent.learning_goals.append("INVESTIGATE: someword")
    harvester.study_cycle()
    assert len(agent.learning_goals) == 0
    print("KnowledgeHarvester: study_cycle resolved an investigation goal.")


def test_refinement_cycle_no_chunky_fact(harvester, agent, monkeypatch):
    # CORRECT: Patch the class
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
    """
    Tests that _deepen_knowledge_of_random_concept successfully adds new
    learning goals by mocking its external dependencies.
    """
    agent.graph.graph.clear()
    agent.learning_goals.clear()
    for name, node_type in graph_nodes:
        agent.graph.add_node(ConceptNode(name=name, node_type=node_type))

    # KEY FIX: Mock the external web search to return a predictable fact.
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_wikipedia",
        lambda self, topic: (topic, f"{topic} is a test concept."),
    )
    # ALSO mock the other web search to do nothing.
    monkeypatch.setattr(
        "axiom.knowledge_harvester.KnowledgeHarvester.get_fact_from_duckduckgo",
        lambda self, topic: None,
    )

    # Mock random.choice to be deterministic
    monkeypatch.setattr("axiom.knowledge_harvester.random.choice", lambda x: x[0])

    harvester._deepen_knowledge_of_random_concept()

    # VERIFICATION: The method doesn't add goals directly. It calls agent.chat().
    # We need to check if the fact was learned in the graph.
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

    # KEY FIX: The name must have MORE THAN 5 words to be "chunky".
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


@pytest.mark.parametrize(
    ("api_response", "should_succeed"),
    [
        # "Happy path": API returns a good definition
        (
            {
                "AbstractText": "A programming language created by Guido van Rossum. It is dynamically typed.",
            },
            True,
        ),
        # "Sad path": API returns an empty response
        ({}, False),
    ],
)
def test_harvester_gets_fact_from_duckduckgo(
    harvester,
    monkeypatch,
    api_response,
    should_succeed,
):
    """
    Covers the get_fact_from_duckduckgo method in the KnowledgeHarvester.
    Tests both a successful API response and a failed/empty response.
    """
    # 1. Setup: Mock the external requests.get call
    topic = "Python"
    mock_response = MockResponse(api_response)

    # When the code calls requests.get, return our fake response object
    monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

    # 2. Action: Call the method we are testing
    result = harvester.get_fact_from_duckduckgo(topic)

    # 3. Verification
    if should_succeed:
        assert result is not None
        result_topic, fact = result
        assert result_topic == topic
        assert fact == "A programming language created by Guido van Rossum."
        print("DuckDuckGo harvester successfully extracted a fact.")
    else:
        assert result is None
        print("DuckDuckGo harvester correctly handled an empty API response.")


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
