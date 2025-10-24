# in tests/test_conversational_flows.py

from __future__ import annotations

from typing import TYPE_CHECKING

from axiom.graph_core import ConceptNode
from axiom.universal_interpreter import InterpretData

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent


def test_agent_handles_stalemate_correctly(agent: CognitiveAgent):
    """
    Test that when a 'Stalemate' conflict occurs, the agent correctly
    triggers clarification AND creates a RESOLVE_CONFLICT learning goal.
    """
    # 1. GIVEN: The words used in the facts are known and trusted.
    agent.lexicon._promote_word_for_test("paris", "noun")
    agent.lexicon._promote_word_for_test("lyon", "noun")
    agent.lexicon._promote_word_for_test("france", "noun")
    agent.lexicon._promote_word_for_test("capital", "noun")

    # 2. WHEN: The agent learns a fact, then a contradictory one of equal confidence.
    agent.chat("Paris is the capital of France")
    clarification_question = agent.chat("Lyon is the capital of France")

    # 3. THEN: Verify the new, unified "Stalemate" behavior.

    # a) It SHOULD be awaiting clarification and should have asked a question.
    assert agent.is_awaiting_clarification is True, (
        "Agent should ask for clarification on a stalemate."
    )
    assert "?" in clarification_question, "Agent should have returned a question."

    # b) It SHOULD have created a specific learning goal for the autonomous system.
    assert len(agent.learning_goals) == 1, "A learning goal should have been created."
    assert "RESOLVE_CONFLICT" in agent.learning_goals[0], (
        "The goal should be to resolve a conflict."
    )
    assert "lyon" in agent.learning_goals[0], (
        "The learning goal should mention the new conflict 'lyon'."
    )
    assert "paris" in agent.learning_goals[0], (
        "The learning goal should mention the existing conflict 'paris'."
    )

    # c) It should NOT have learned the second fact yet.
    france_node = agent.graph.get_node_by_name("france")
    assert france_node is not None

    capitals = []
    for edge in agent.graph.get_edges_from_node(france_node.id):
        if edge.type == "has_capital":
            target_node = agent.graph.get_node_by_id(edge.target)
            if target_node:
                capitals.append(target_node.name)

    assert capitals == ["paris"], (
        "Agent should not learn the conflicting fact until clarified."
    )

    print(
        "Agent correctly handled stalemate by asking for clarification and creating a learning goal.",
    )


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
