# in tests/test_conversational_flows.py

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from axiom.graph_core import ConceptNode
from axiom.universal_interpreter import InterpretData

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent


def test_agent_diverts_to_clarification_handler_when_awaiting(
    agent: CognitiveAgent,
    monkeypatch,
):
    """
    Covers the 'if self.is_awaiting_clarification:' branch in the chat method.
    Ensures that when the agent is in this state, the input is correctly
    routed to the _handle_clarification method.
    """
    # 1. GIVEN: Put the agent into an awaiting clarification state.
    agent.chat("Paris is the capital of France")
    clarification_question = agent.chat("Lyon is the capital of France")
    assert agent.is_awaiting_clarification is True
    assert "?" in clarification_question, "Agent should have asked a question."

    # 2. MOCK: Replace the real _handle_clarification method with a "spy".
    mock_handler = MagicMock()
    monkeypatch.setattr(agent, "_handle_clarification", mock_handler)

    mock_normal_flow_spy = MagicMock()
    monkeypatch.setattr(agent, "_expand_contractions", mock_normal_flow_spy)

    # 3. WHEN: The user provides an answer to the clarification question.
    user_answer = "Paris"
    agent.chat(user_answer)

    # 4. THEN: Verify the correct path was taken.
    mock_handler.assert_called_once_with(user_answer)
    mock_normal_flow_spy.assert_not_called()
    print("Agent correctly diverted input to the clarification handler.")


def test_agent_handle_clarification_reinforces_and_punishes(
    agent: CognitiveAgent,
    monkeypatch,
):
    """
    Covers the internal logic of the _handle_clarification method with a true unit test.
    """
    # 1. GIVEN: Manually set up the agent's state.
    # a) Create the conflicting facts directly in the graph.
    france_node = agent.graph.add_node(ConceptNode(name="france"))
    paris_node = agent.graph.add_node(ConceptNode(name="paris"))
    lyon_node = agent.graph.add_node(ConceptNode(name="lyon"))

    # Add both edges. Give them an initial weight.
    agent.graph.add_edge(france_node, paris_node, "is_capital_of", weight=0.5)
    agent.graph.add_edge(france_node, lyon_node, "is_capital_of", weight=0.5)

    # b) Manually set the agent's internal state to be awaiting clarification.
    agent.is_awaiting_clarification = True
    agent.clarification_context = {
        "subject": "france",
        "conflicting_relation": "is_capital_of",
        "conflicting_nodes": ["paris", "lyon"],
    }

    # 2. MOCK: Mock the interpreter's response to the user's clarification.
    # It will correctly identify "Paris" as the entity in the user's answer.
    mock_interpretation = InterpretData(
        intent="statement_of_fact",
        entities=[{"name": "Paris", "type": "CONCEPT"}],
        relation=None,
        key_topics=["Paris"],
        full_text_rephrased="Paris",
    )
    monkeypatch.setattr(
        agent.interpreter,
        "interpret",
        lambda *args, **kwargs: mock_interpretation,
    )

    # 3. WHEN: We call the method under test directly.
    response = agent._handle_clarification("Paris is the correct one")

    # 4. THEN: Verify the logic of _handle_clarification.
    assert agent.is_awaiting_clarification is False
    assert "Thank you for the clarification" in response

    # Inspect the knowledge graph to verify the weights were updated.
    paris_edge_weight = -1.0
    lyon_edge_weight = -1.0
    for edge in agent.graph.get_edges_from_node(france_node.id):
        if edge.type == "is_capital_of":
            target_node = agent.graph.get_node_by_id(edge.target)
            if target_node and target_node.name == "paris":
                paris_edge_weight = edge.weight
            elif target_node and target_node.name == "lyon":
                lyon_edge_weight = edge.weight

    assert paris_edge_weight == 1.0, "The correct fact should have been reinforced."
    assert lyon_edge_weight == 0.1, "The incorrect fact should have been punished."

    print("Agent correctly handled clarification, reinforcing and punishing facts.")
