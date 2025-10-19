from __future__ import annotations

from unittest.mock import MagicMock

# tests/test_initialization.py
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
    # 1. Setup: Create a valid, pre-seeded brain file.
    brain_file = tmp_path / "good_brain.json"
    state_file = tmp_path / "good_state.json"

    # Create a temporary graph and give it the agent's identity
    temp_graph = ConceptGraph()
    agent_node = temp_graph.add_node(ConceptNode(name="agent"))
    name_node = temp_graph.add_node(ConceptNode(name="TestAgent"))
    temp_graph.add_edge(agent_node, name_node, "has_name")

    # Save this "good" brain to the file
    temp_graph.save_to_file(brain_file)

    # 2. Mock: We need to prevent the real UniversalInterpreter from loading.
    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())

    # We also "spy" on the seeding function to prove it's NOT called.
    seed_was_called = False

    def mock_seed_knowledge(*args, **kwargs):
        nonlocal seed_was_called
        seed_was_called = True

    monkeypatch.setattr(
        "axiom.cognitive_agent.seed_domain_knowledge",
        mock_seed_knowledge,
    )

    # 3. Action: Initialize the agent from our pre-made "good" brain file.
    agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    # 4. Verification:
    # Prove that the health check passed by confirming the brain was NOT re-seeded.
    assert seed_was_called is False

    # Prove that the specific lines were run by checking the graph state.
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
    # 1. Setup: Create a broken/corrupt state file.
    # We'll just write some non-JSON text to it.
    brain_file = tmp_path / "good_brain.json"  # Needs a brain file to exist
    state_file = tmp_path / "corrupt_state.json"
    state_file.write_text("this is not valid json")

    # We need a valid brain file for the main health check to pass.
    temp_graph = ConceptGraph()
    agent_node = temp_graph.add_node(ConceptNode(name="agent"))
    name_node = temp_graph.add_node(ConceptNode(name="TestAgent"))
    temp_graph.add_edge(agent_node, name_node, "has_name")
    temp_graph.save_to_file(brain_file)

    # Mock the interpreter to prevent LLM loading
    monkeypatch.setattr("axiom.cognitive_agent.UniversalInterpreter", MagicMock())

    # 2. Action: Initialize the agent, pointing it to the corrupt state file.
    # This should NOT crash.
    agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    # 3. Verification:
    # Check that the agent correctly reset its state to the default.
    assert agent.learning_iterations == 0
    print("Agent successfully handled a corrupt state file and reset its state.")


def test_agent_reseeds_corrupt_brain_file(monkeypatch, tmp_path):
    """
    Covers the __init__ "health check" failure path for a corrupt brain file.
    Ensures that the brain IS re-seeded if the agent's core identity is missing.
    """
    # 1. Setup: Create a "corrupt" brain file. It's valid JSON, but
    # it's missing the crucial 'has_name' relationship.
    brain_file = tmp_path / "corrupt_brain.json"
    state_file = tmp_path / "corrupt_state.json"

    # Create a temporary graph that has an 'agent' node but no name.
    temp_graph = ConceptGraph()
    temp_graph.add_node(ConceptNode(name="agent"))
    temp_graph.save_to_file(brain_file)

    # 2. Mock: We don't want to run the actual slow seeding process.
    # Instead, we create a "spy" (a MagicMock) to watch if the seeding
    # functions are called.
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

    # 3. Action: Initialize the agent from our pre-made "corrupt" brain file.
    _ = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    # 4. Verification:
    # Prove that the health check failed by asserting that the seeding functions
    # were called exactly once.
    mock_seed_domain.assert_called_once()
    mock_seed_vocab.assert_called_once()
    print("Agent correctly detected a corrupt brain and triggered a re-seed.")
