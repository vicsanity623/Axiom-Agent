from __future__ import annotations

# tests/test_initialization.py
import pytest

from axiom.cognitive_agent import CognitiveAgent


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