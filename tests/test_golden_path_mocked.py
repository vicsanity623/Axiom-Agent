"""
CI-safe golden path test for Axiom-Agent.

Mocks the LLM fallback so this can run without any model downloads.
Demonstrates the same learn → query → introspect → recall loop deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axiom.cognitive_agent import CognitiveAgent


@pytest.fixture
def fresh_agent(monkeypatch):
    """Creates a clean agent with mocked LLM fallback."""
    brain_file = Path("brain/test_golden_path_mock.json")
    state_file = Path("brain/test_golden_path_mock_state.json")
    brain_file.unlink(missing_ok=True)
    state_file.unlink(missing_ok=True)

    agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)

    # ---- Mock the LLM fallback deterministically ----
    def fake_llm_response(prompt: str) -> str:
        # always leak that "raven is black" when asked color
        if "color" in prompt.lower():
            return "Ravens are black birds."
        return "I'm a mock model."

    agent.llm_generate = fake_llm_response

    yield agent

    # cleanup (optional)
    brain_file.unlink(missing_ok=True)
    state_file.unlink(missing_ok=True)


def test_golden_path_learning_and_introspection_mocked(fresh_agent):
    agent = fresh_agent

    # 1. Teach the agent a fact.
    resp1 = agent.chat("a raven is a bird")
    assert "I understand" in resp1

    # verify knowledge stored
    raven = agent.graph.get_node_by_name("raven")
    bird = agent.graph.get_node_by_name("bird")
    edges = [
        agent.graph.get_node_by_id(e.target).name
        for e in agent.graph.get_edges_from_node(raven.id)
    ]
    assert bird.name in edges

    # 2. Ask unknown question -> triggers mocked LLM fallback
    resp2 = agent.chat("what color is it?")
    assert "black" in resp2.lower()

    # 3. Check that introspection worked (symbolic learning)
    edges2 = [
        agent.graph.get_node_by_id(e.target).name
        for e in agent.graph.get_edges_from_node(raven.id)
    ]
    assert "black" in edges2

    # 4. Ask again -> should answer from knowledge base (no fallback)
    agent.llm_generate = lambda _: "SHOULD NOT BE CALLED"
    resp3 = agent.chat("what color is it?")
    assert "black" in resp3.lower()
