# in tests/test_core_functionality.py

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from axiom.cognitive_agent import CognitiveAgent

# --- Test Setup: A "Stunt Double" for the LLM Interpreter ---


class MockUniversalInterpreter:
    """
    A fake UniversalInterpreter that does NOTHING related to LLMs.
    It allows us to create a CognitiveAgent without it trying to load a model file.
    """

    def __init__(self, *args, **kwargs):
        # This __init__ method is empty on purpose.
        print("--- Initialized MockUniversalInterpreter (No LLM Loaded) ---")
        pass


# --- A Reusable Agent for All Our Tests ---


@pytest.fixture
def agent(monkeypatch, tmp_path: Path) -> CognitiveAgent:
    """
    This is a Pytest "fixture". It creates a fresh, clean CognitiveAgent
    for every single test that needs one. It automatically uses our mock
    interpreter so we never have to worry about the LLM.
    """
    # Use monkeypatch to replace the real, slow interpreter with our fast, fake one.
    monkeypatch.setattr(
        "axiom.cognitive_agent.UniversalInterpreter",
        MockUniversalInterpreter,
    )

    # Create temporary brain files for a clean test environment
    brain_file = tmp_path / "test_brain.json"
    state_file = tmp_path / "test_state.json"

    # Create and return the agent. The LLM is disabled because of the mock.
    return CognitiveAgent(brain_file=brain_file, state_file=state_file)


# --- The Actual Tests ---


def test_agent_initialization(agent: CognitiveAgent):
    """Tests that the agent can be created successfully without errors."""
    assert agent is not None
    assert isinstance(agent, CognitiveAgent)
    print("Agent initialized successfully.")


def test_learning_a_fact(agent: CognitiveAgent):
    """Tests that the agent can learn a simple fact and store it in its graph."""
    # 1. Action: Teach the agent a new fact.
    response = agent.chat("a horse is an animal")
    assert "I understand" in response

    # 2. Verification: Check the agent's brain to see if it learned correctly.
    fact_is_known = False
    horse_node = agent.graph.get_node_by_name("horse")
    assert horse_node is not None, "Agent did not create a node for 'horse'."

    for edge in agent.graph.get_edges_from_node(horse_node.id):
        target_node = agent.graph.get_node_by_id(edge.target)
        if edge.type == "is_a" and target_node and target_node.name == "animal":
            fact_is_known = True
            break

    assert fact_is_known is True, "Agent failed to learn 'horse is an animal'."
    print("Agent successfully learned a fact.")


@pytest.mark.parametrize(
    ("sentence", "expected_subject", "expected_relation", "expected_object"),
    [
        ("Paris is a city in France", "paris", "is_located_in", "france"),
        ("a wheel is part of a car", "wheel", "is_part_of", "car"),
    ],
)
def test_parser_handles_prepositions(
    agent: CognitiveAgent,
    sentence,
    expected_subject,
    expected_relation,
    expected_object,
):
    """Tests that the symbolic parser can handle different prepositional phrases."""
    # 1. Action: Parse the sentence
    interpretations = agent.parser.parse(sentence)
    assert interpretations, "Parser failed to produce an interpretation."

    relation = interpretations[0].get("relation")
    assert relation, "Parser failed to extract a relation."

    # 2. Verification: Check if the components are correct
    assert relation.get("subject") == expected_subject
    assert relation.get("verb") == expected_relation
    assert relation.get("object") == expected_object
    print(f"Parser correctly handled: '{sentence}'")
