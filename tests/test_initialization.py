from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from axiom.cognitive_agent import CognitiveAgent, ConceptGraph, ConceptNode


def test_agent_initialization_from_data() -> None:
    """Test the successful initialization of the agent from data dictionaries.

    This test verifies that the CognitiveAgent can be created in
    inference-only mode by passing in empty brain and cache data. It serves
    as a basic "smoke test" to ensure the core class and its sub-components
    (interpreters, parsers) can be instantiated without crashing.
    """
    agent = CognitiveAgent(
        load_from_file=False,
        brain_data={"nodes": [], "links": []},
        cache_data={"interpretations": [], "synthesis": []},
        inference_mode=True,
        enable_llm=False,
    )
    assert agent is not None
    assert agent.inference_mode is True
    assert agent.graph is not None
    assert agent.interpreter is not None
    assert agent.interpreter.llm is None
    assert agent.parser is not None
    assert agent.lexicon is not None


def test_agent_initialization_no_data_fails() -> None:
    """Test that agent initialization fails if no data or files are provided."""
    with pytest.raises(ValueError, match="Agent must be initialized"):
        _ = CognitiveAgent(
            load_from_file=False,
            enable_llm=False,
        )


def test_agent_initialization_from_valid_brain_file(monkeypatch, tmp_path):
    """
    Covers the __init__ "health check" path where a valid brain file is loaded.
    Ensures that the brain is NOT re-seeded if the agent's identity is present.
    """
    brain_file = tmp_path / "good_brain.json"
    state_file = tmp_path / "good_state.json"

    temp_graph = ConceptGraph()
    agent_node = temp_graph.add_node(ConceptNode(name="agent"))
    name_node = temp_graph.add_node(ConceptNode(name="TestAgent"))
    temp_graph.add_edge(agent_node, name_node, "has_name")

    temp_graph.save_to_file(brain_file)

    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())

    seed_was_called = False

    def mock_seed_knowledge(*args, **kwargs):
        nonlocal seed_was_called
        seed_was_called = True

    monkeypatch.setattr(
        "axiom.cognitive_agent.seed_domain_knowledge",
        mock_seed_knowledge,
    )

    agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    assert seed_was_called is False

    loaded_agent_node = agent.graph.get_node_by_name("agent")
    assert loaded_agent_node is not None

    name_edge = next(
        (
            edge
            for edge in agent.graph.get_edges_from_node(loaded_agent_node.id)
            if edge.type == "has_name"
        ),
        None,
    )
    assert name_edge is not None
    print("Agent successfully loaded from a valid brain file without re-seeding.")


def test_agent_handles_corrupt_state_file(monkeypatch, tmp_path):
    """
    Covers the exception handling in the _load_agent_state method.
    Ensures the agent can start up with a fresh state if the old one is broken.
    """
    brain_file = tmp_path / "good_brain.json"
    state_file = tmp_path / "corrupt_state.json"
    state_file.write_text("this is not valid json")

    temp_graph = ConceptGraph()
    agent_node = temp_graph.add_node(ConceptNode(name="agent"))
    name_node = temp_graph.add_node(ConceptNode(name="TestAgent"))
    temp_graph.add_edge(agent_node, name_node, "has_name")
    temp_graph.save_to_file(brain_file)

    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())

    agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    assert agent.learning_iterations == 0
    print("Agent successfully handled a corrupt state file and reset its state.")


def test_agent_reseeds_corrupt_brain_file(monkeypatch, tmp_path):
    """
    Covers the __init__ "health check" failure path for a corrupt brain file.
    Ensures that the brain IS re-seeded if the agent's core identity is missing.
    """
    brain_file = tmp_path / "corrupt_brain.json"
    state_file = tmp_path / "corrupt_state.json"

    temp_graph = ConceptGraph()
    temp_graph.add_node(ConceptNode(name="agent"))
    temp_graph.save_to_file(brain_file)

    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())
    mock_seed_domain = MagicMock()
    mock_seed_vocab = MagicMock()
    monkeypatch.setattr(
        "axiom.cognitive_agent.seed_domain_knowledge",
        mock_seed_domain,
    )
    monkeypatch.setattr(
        "axiom.cognitive_agent.seed_core_vocabulary",
        mock_seed_vocab,
    )

    _ = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    mock_seed_domain.assert_called_once()
    mock_seed_vocab.assert_called_once()
    print("Agent correctly detected a corrupt brain and triggered a re-seed.")
