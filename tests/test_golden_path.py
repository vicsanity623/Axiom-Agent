# in tests/test_golden_path.py

from __future__ import annotations

from pathlib import Path

import pytest

from axiom.cognitive_agent import CognitiveAgent

# This test requires the LLM model to be present to test the introspection loop.
# We can add markers to skip it if the model is not found later.


@pytest.mark.introspection
def test_golden_path_learning_and_introspection():
    """
    Run a full, end-to-end "golden path" test of the agent's lifecycle:
    1. Learns a new fact.
    2. Answers a question, triggering LLM synthesis.
    3. Introspectively learns from its own synthesized response.
    4. Answers the same question again, now with the new knowledge.
    """
    # 1. Setup: Create a clean brain for a reproducible test.
    brain_file = Path("brain/test_golden_path.json")
    state_file = Path("brain/test_golden_path_state.json")
    if brain_file.exists():
        brain_file.unlink()
    if state_file.exists():
        state_file.unlink()

    agent = CognitiveAgent(
        brain_file=brain_file,
        state_file=state_file,
    )

    # 2. Turn 1: Teach the agent a new, novel fact.
    response1 = agent.chat("a raven is a bird")
    assert "I understand" in response1

    # Verify that the fact was learned.
    fact_is_known = False
    raven_node = agent.graph.get_node_by_name("raven")
    if raven_node:
        for edge in agent.graph.get_edges_from_node(raven_node.id):
            target_node = agent.graph.get_node_by_id(edge.target)
            if edge.type == "is_a" and target_node and target_node.name == "bird":
                fact_is_known = True
    assert fact_is_known is True, (
        "Agent failed to learn the initial fact 'raven is a bird'"
    )

    # 3. Turn 2: Ask a question that requires LLM synthesis and triggers introspection.
    # The agent's symbolic brain does NOT know the color of a raven.
    response2 = agent.chat("what color is it?")
    # The LLM "leaks" the fact that ravens are black.
    assert "black" in response2.lower()

    # 4. Verification: Check if the introspective learning worked.
    # The agent should have parsed its own response and learned the new fact.
    fact_is_now_known = False
    raven_node = agent.graph.get_node_by_name("raven")
    if raven_node:
        for edge in agent.graph.get_edges_from_node(raven_node.id):
            target_node = agent.graph.get_node_by_id(edge.target)
            if (
                edge.type == "has_property"
                and target_node
                and target_node.name == "black"
            ):
                fact_is_now_known = True
    assert fact_is_now_known is True, (
        "Agent failed to introspectively learn the color of a raven"
    )

    # 5. Turn 3: Ask the same question again.
    # This time, the agent should answer from its own, now-complete symbolic knowledge.
    response3 = agent.chat("what color is it?")
    assert "black" in response3.lower()
