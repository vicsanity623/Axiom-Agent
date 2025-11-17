from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from axiom.cognitive_agent import CognitiveAgent, ConceptGraph, ConceptNode

if TYPE_CHECKING:
    from pathlib import Path


def test_agent_initialization_from_data(tmp_path: Path) -> None:
    """
    Given: Dictionaries representing brain and cache data.
    When: The CognitiveAgent is initialized in inference mode from this data.
    Then: The agent and its components should be instantiated correctly.
    """
    # Arrange
    brain_data: dict[str, Any] = {
        "nodes": [
            {"id": "agent_id", "name": "agent", "type": "core_identity"},
            {"id": "name_id", "name": "TestAgent", "type": "property"},
        ],
        "links": [{"source": "agent_id", "target": "name_id", "type": "has_name"}],
    }
    cache_data: dict[str, Any] = {
        "interpretations": [["key1", {}]],
        "synthesis": [["key2", "value"]],
    }

    mock_interpreter = MagicMock()
    mock_interpreter.interpretation_cache = {}
    mock_interpreter.synthesis_cache = {}
    mock_interpreter.llm = None

    brain_file = tmp_path / "test_brain.json"
    state_file = tmp_path / "test_state.json"

    # Act
    with patch(
        "axiom.cognitive_agent.UniversalInterpreter", return_value=mock_interpreter
    ):
        agent = CognitiveAgent(
            brain_file=brain_file,
            state_file=state_file,
            load_from_file=False,
            brain_data=brain_data,
            cache_data=cache_data,
            inference_mode=True,
            enable_llm=False,
        )

    # Assert
    assert agent is not None
    assert agent.inference_mode is True
    assert agent.graph is not None
    assert agent.interpreter is not None
    assert agent.interpreter.llm is None
    assert "key1" in mock_interpreter.interpretation_cache
    assert "key2" in mock_interpreter.synthesis_cache


def test_agent_initialization_no_data_fails() -> None:
    """
    Given: A call to initialize the agent with load_from_file=False.
    When: No brain_data is provided.
    Then: A ValueError should be raised.
    """
    with pytest.raises(
        ValueError,
        match="Agent must be initialized with either files or data if load_from_file is False.",
    ):
        _ = CognitiveAgent(
            load_from_file=False,
            brain_data=None,
            enable_llm=False,
        )


def test_agent_initialization_from_valid_brain_file(monkeypatch, tmp_path):
    """
    Given: A valid brain file with the agent's identity.
    When: The agent is initialized from this file.
    Then: The brain should be loaded correctly, and re-seeding should NOT be triggered.
    """
    # Arrange
    brain_file = tmp_path / "good_brain.json"
    state_file = tmp_path / "good_state.json"
    state_file.touch()

    temp_graph = ConceptGraph()
    agent_node = temp_graph.add_node(ConceptNode(name="agent"))
    name_node = temp_graph.add_node(ConceptNode(name="TestAgent"))
    temp_graph.add_edge(agent_node, name_node, "has_name")
    temp_graph.save_to_file(brain_file)

    mock_seed_domain = MagicMock()
    monkeypatch.setattr("axiom.cognitive_agent.seed_domain_knowledge", mock_seed_domain)
    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())

    # Act
    agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)
    agent.load_state()  # Simulate the full startup sequence

    # Assert
    mock_seed_domain.assert_not_called()
    loaded_agent_node = agent.graph.get_node_by_name("agent")
    assert loaded_agent_node is not None


def test_agent_handles_corrupt_state_file(monkeypatch, tmp_path):
    """
    Given: A corrupt (invalid JSON) state file.
    When: The agent's public load_state() method is called.
    Then: The agent should handle the error gracefully and start with a fresh, empty state.
    """
    # Arrange
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "corrupt_state.json"
    state_file.write_text("this is not valid json")
    brain_file.touch()

    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())

    # Initialize the agent (it will have an empty brain and state)
    agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    # Pre-populate state to ensure it gets cleared
    agent.learning_goals.append("some old goal")

    # Act
    # This call should catch the JSONDecodeError and reset the state attributes.
    agent.load_state()

    # Assert
    assert agent.learning_goals == []
    assert agent.high_priority_tasks == []
    assert agent.autonomous_cycle_count == 0


def test_agent_reseeds_corrupt_brain_file(monkeypatch, tmp_path):
    """
    Given: A brain file that is structurally valid but fails the integrity check.
    When: The CognitiveAgent is initialized.
    Then: The re-seeding process should be triggered.
    """
    # Arrange
    brain_file = tmp_path / "corrupt_brain.json"
    state_file = tmp_path / "state.json"
    state_file.touch()

    # This brain is "corrupt" because it's missing the required 'has_name' edge.
    corrupt_brain_data = {
        "nodes": [{"id": "agent", "name": "agent", "type": "core_identity"}],
        "links": [],
    }
    brain_file.write_text(json.dumps(corrupt_brain_data))

    mock_seed_domain = MagicMock()
    mock_seed_vocab = MagicMock()
    monkeypatch.setattr("axiom.cognitive_agent.seed_domain_knowledge", mock_seed_domain)
    monkeypatch.setattr("axiom.cognitive_agent.seed_core_vocabulary", mock_seed_vocab)
    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())
    monkeypatch.setattr("axiom.cognitive_agent.CognitiveAgent.save_brain", MagicMock())
    monkeypatch.setattr("axiom.cognitive_agent.CognitiveAgent.save_state", MagicMock())

    # Act
    _ = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    # Assert
    mock_seed_domain.assert_called_once()
    mock_seed_vocab.assert_called_once()
