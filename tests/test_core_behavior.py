import sys
from pathlib import Path
from typing import cast

import pytest

# Ensure src is in sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_base import validate_and_add_relation
from axiom.universal_interpreter import PropertyData, RelationData


# Test helper fixture
@pytest.fixture
def agent(tmp_path: Path) -> CognitiveAgent:
    """Provides a fresh, non-learning CognitiveAgent for testing."""
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "state.json"
    return CognitiveAgent(
        brain_file=brain_file,
        state_file=state_file,
        load_from_file=True,
        enable_llm=False,
    )


def test_process_statement_for_learning_success(agent: CognitiveAgent):
    """
    Tests the happy path: a valid statement with known vocabulary is learned correctly.
    """
    # Arrange: Use unique, novel concepts to ensure the fact does not already exist.
    subject_word = "axolotl"
    object_word = "amphibian"

    # Explicitly promote the words to satisfy the vocabulary check.
    agent.lexicon.promote_word(subject_word, "noun")
    agent.lexicon.promote_word(object_word, "noun")

    relation = RelationData(
        subject=subject_word,
        verb="is_a",
        object=object_word,
        properties=cast("PropertyData", {"confidence": 0.9, "provenance": "user"}),
    )

    # Act
    was_learned, message = agent._process_statement_for_learning(relation)

    # Assert
    assert was_learned is True, f"Learning failed unexpectedly with message: {message}"
    assert message == "I understand. I have noted that."

    src_node = agent.graph.get_node_by_name(subject_word)
    tgt_node = agent.graph.get_node_by_name(object_word)
    assert src_node is not None
    assert tgt_node is not None

    edges = agent.graph.get_edges_from_node(src_node.id)
    assert any(e.target == tgt_node.id and e.type == "is_a" for e in edges)


def test_validate_and_add_relation_defers_unknown_words(agent: CognitiveAgent):
    """
    Given a relation containing unknown words, validate_and_add_relation
    should return 'deferred' and create new INVESTIGATE goals.
    """
    # Arrange
    assert not agent.lexicon.is_known_word("newwordify")

    relation_dict = {
        "subject": "newwordify",
        "verb": "is_a",
        "object": "flangdoodle",
    }
    properties = cast("PropertyData", {"confidence": 0.3, "provenance": "user"})

    # Act
    status = validate_and_add_relation(agent, relation_dict, properties)

    # Assert
    assert status == "deferred"
    assert "INVESTIGATE: newwordify" in agent.learning_goals
    assert "INVESTIGATE: flangdoodle" in agent.learning_goals


def test_belief_revision_rejects_weaker_fact(agent: CognitiveAgent):
    """
    Tests that the agent correctly rejects a new, weaker fact that conflicts
    with a stronger, existing exclusive fact.
    """
    # Arrange: Teach a high-confidence "seed" fact.
    agent.lexicon.promote_word("france", "noun")  # Promote words for this test
    agent.lexicon.promote_word("paris", "noun")
    agent.lexicon.promote_word("lyon", "noun")

    initial_fact = RelationData(
        subject="france",
        verb="has_capital",
        object="paris",
        properties=cast("PropertyData", {"confidence": 1.0, "provenance": "seed"}),
    )
    agent._process_statement_for_learning(initial_fact)

    # Act: Attempt to learn a weaker, conflicting fact.
    conflicting_fact = RelationData(
        subject="france",
        verb="has_capital",
        object="lyon",
        properties=cast("PropertyData", {"confidence": 0.8, "provenance": "user"}),
    )
    was_learned, message = agent._process_statement_for_learning(conflicting_fact)

    # Assert: The new fact was not learned, and the correct reason was given.
    assert was_learned is False
    assert message == "existing_fact_stronger"


def test_belief_revision_handles_stalemate(agent: CognitiveAgent):
    """
    Tests that learning a conflicting fact of equal strength triggers a
    stalemate and signals for clarification.
    """
    # Arrange: Teach a user-provided fact.
    agent.lexicon.promote_word("france", "noun")  # Promote words for this test
    agent.lexicon.promote_word("paris", "noun")
    agent.lexicon.promote_word("lyon", "noun")

    initial_fact = RelationData(
        subject="france",
        verb="has_capital",
        object="paris",
        properties=cast("PropertyData", {"confidence": 0.8, "provenance": "user"}),
    )
    agent._process_statement_for_learning(initial_fact)

    # Act: Attempt to learn a conflicting fact of the exact same strength.
    conflicting_fact = RelationData(
        subject="france",
        verb="has_capital",
        object="lyon",
        properties=cast("PropertyData", {"confidence": 0.8, "provenance": "user"}),
    )
    was_learned, message = agent._process_statement_for_learning(conflicting_fact)

    # Assert: The new fact was not learned, and the stalemate was correctly identified.
    assert was_learned is False
    assert message == "exclusive_conflict"
