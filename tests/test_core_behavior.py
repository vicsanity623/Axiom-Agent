from pathlib import Path
from typing import cast

import pytest

from axiom.cognitive_agent import CognitiveAgent
from axiom.graph_core import ConceptGraph, ConceptNode

# --- FIX: Import the correct 'Task' type and remove the incorrect 'Goal' import ---
from axiom.universal_interpreter import RelationData, Task


@pytest.fixture
def graph() -> ConceptGraph:
    """Provide a fresh ConceptGraph for testing."""
    return ConceptGraph()


def test_add_and_get_node(graph: ConceptGraph):
    node = ConceptNode("Dog", node_type="noun")
    graph.add_node(node)
    retrieved = graph.get_node_by_name("dog")
    assert retrieved is not None
    assert retrieved.name == "dog"
    assert retrieved.type == "noun"


def test_add_and_get_edge(graph: ConceptGraph):
    src = graph.add_node(ConceptNode("Cat"))
    tgt = graph.add_node(ConceptNode("Mammal"))
    edge = graph.add_edge(src, tgt, "is_a", weight=0.9)
    assert edge is not None
    edges_from_src = graph.get_edges_from_node(src.id)
    assert any(e.target == tgt.id and e.type == "is_a" for e in edges_from_src)


def test_update_edge_properties(graph: ConceptGraph):
    src = graph.add_node(ConceptNode("Car"))
    tgt = graph.add_node(ConceptNode("Vehicle"))
    edge = graph.add_edge(src, tgt, "is_a", weight=0.5)
    assert edge is not None
    graph.update_edge_properties(edge, {"confidence": 0.8})
    updated_edge = graph.get_edges_from_node(src.id)[0]
    assert updated_edge.properties["confidence"] == 0.8


def test_update_edge_weight(graph: ConceptGraph):
    src = graph.add_node(ConceptNode("Apple"))
    tgt = graph.add_node(ConceptNode("Fruit"))
    edge = graph.add_edge(src, tgt, "is_a", weight=0.5)
    assert edge is not None
    graph.update_edge_weight(edge, new_weight=0.9, provenance="system")
    updated_edge = graph.get_edges_from_node(src.id)[0]
    assert updated_edge.weight > 0.5


def test_conflicting_edge_detection(graph: ConceptGraph):
    src = graph.add_node(ConceptNode("Paris"))
    tgt1 = graph.add_node(ConceptNode("France"))
    tgt2 = graph.add_node(ConceptNode("Germany"))

    edge1 = graph.add_edge(src, tgt1, "capital_of", weight=1.0)
    edge2 = graph.add_edge(src, tgt2, "capital_of", weight=0.8)
    assert edge1 is not None
    assert edge2 is not None

    winner = graph.find_exclusive_conflict(src, "capital_of")
    assert winner is not None
    graph.update_edge_properties(winner, {"example": True})

    assert winner.id == edge1.id
    assert winner.id != edge2.id


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


def test_validate_and_add_relation_defers_unknown_words(agent: CognitiveAgent):
    """
    Given a relation containing unknown words, the learning process should
    return 'deferred' and create new structured INVESTIGATE tasks in the
    high-priority queue.
    """
    # 1. SETUP: Ensure the words are not known.
    assert not agent.lexicon.is_known_word("newwordify")
    assert not agent.lexicon.is_known_word("flangdoodle")

    # 2. ARRANGE: Create the raw relation data.
    relation_dict = cast(
        "RelationData",
        {
            "subject": "newwordify",
            "verb": "is a",
            "object": "flangdoodle",
            "properties": {"confidence": 0.3, "provenance": "user"},
        },
    )

    # 3. ACT: Call the high-level learning method, which is responsible for the full pipeline.
    # We are testing the behavior of the entire learning process for a single statement.
    success, message = agent.knowledge_manager._process_statement_for_learning(
        relation_dict
    )

    # 4. ASSERT: Verify the deferral behavior.
    assert success is False
    assert message == "deferred"

    expected_task_1 = Task(
        task="INVESTIGATE",
        topic="newwordify",
        context="related to the concept of 'newwordify'",
    )
    expected_task_2 = Task(
        task="INVESTIGATE",
        topic="flangdoodle",
        context="related to the concept of 'newwordify'",
    )

    # Check that the tasks were created and added to the high-priority queue.
    assert expected_task_1 in agent.high_priority_tasks
    assert expected_task_2 in agent.high_priority_tasks
    assert len(agent.learning_goals) == 0
