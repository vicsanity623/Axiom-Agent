from unittest.mock import MagicMock

import pytest

from axiom.cognitive_agent import CognitiveAgent
from axiom.graph_core import ConceptNode, RelationshipEdge
from axiom.symbolic_synthesizer import SymbolicSynthesizer


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=CognitiveAgent)
    agent.graph = MagicMock()
    return agent


@pytest.fixture
def synthesizer(mock_agent):
    return SymbolicSynthesizer(mock_agent)


def test_synthesize_simple_svo(synthesizer, mock_agent):
    # Setup graph nodes
    subject = ConceptNode("cat")
    object_node = ConceptNode("mammal")
    mock_agent.graph.get_node_by_id.side_effect = (
        lambda x: subject if x == "s1" else object_node
    )

    # Create edge
    edge = RelationshipEdge(source="s1", target="o1", type="is_a")

    # Test synthesis
    facts = [("cat is_a mammal", edge)]
    result = synthesizer.synthesize(facts, "cat")

    assert result is not None
    # Check for any of the possible templates
    possible_outputs = [
        "Cat is a type of mammal.",
        "Cat is a mammal.",
        "Cat falls under the category of mammal.",
    ]
    assert any(expected in result for expected in possible_outputs)


def test_synthesize_multiple_facts(synthesizer, mock_agent):
    # Setup graph nodes
    subject = ConceptNode("dog")
    obj1 = ConceptNode("animal")
    obj2 = ConceptNode("bark")

    def get_node(nid):
        if nid == "s1":
            return subject
        if nid == "o1":
            return obj1
        if nid == "o2":
            return obj2
        return None

    mock_agent.graph.get_node_by_id.side_effect = get_node

    edge1 = RelationshipEdge(source="s1", target="o1", type="is_a")
    edge2 = RelationshipEdge(source="s1", target="o2", type="capable_of")

    facts = [("", edge1), ("", edge2)]
    result = synthesizer.synthesize(facts, "dog")

    assert result is not None
    assert "Dog" in result
    assert "animal" in result
    assert "bark" in result


def test_temporal_filtering(mock_agent):
    # We need to test the ConversationManager logic, but we can mock it or import it
    from axiom.conversation_manager import ConversationManager

    cm = ConversationManager(mock_agent)

    # Setup facts with dates
    # Fact 1: CEO in 2010
    props1 = (("effective_date", "2010-01-01"),)
    fact1 = ("Obama was president", props1)

    # Fact 2: CEO in 2020
    props2 = (("effective_date", "2020-01-01"),)
    fact2 = ("Trump was president", props2)

    facts = (fact1, fact2)

    # Test 1: Query for 2010
    filtered_2010 = cm._filter_facts_for_temporal_query(
        facts, "Who was president in 2010?"
    )
    assert len(filtered_2010) == 1
    assert "Obama" in list(filtered_2010)[0]

    # Test 2: Query for 2020
    filtered_2020 = cm._filter_facts_for_temporal_query(
        facts, "Who was president in 2020?"
    )
    assert len(filtered_2020) == 1
    assert "Trump" in list(filtered_2020)[0]
