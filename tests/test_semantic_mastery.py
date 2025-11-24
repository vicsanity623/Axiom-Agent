from unittest.mock import MagicMock

import pytest

from axiom.cognitive_agent import CognitiveAgent
from axiom.graph_core import ConceptNode


@pytest.fixture
def agent():
    # Initialize agent in inference mode to avoid side effects, but enable LLM mock
    agent = CognitiveAgent(inference_mode=False, enable_llm=False)
    # Mock the interpreter to avoid actual LLM calls
    agent.interpreter = MagicMock()
    agent.interpreter.disambiguate_word.return_value = None
    return agent


def test_homonym_creation(agent):
    """Test that teaching two different meanings creates two sense nodes."""

    # 1. Teach "A bank is a financial institution"
    # Mock disambiguation for the first case
    agent.interpreter.disambiguate_word.return_value = "financial institution"

    # Manually trigger the learning process
    # We simulate the flow: sentence -> parser/interpreter -> knowledge_manager

    # Create the base node first
    base_node = agent.knowledge_manager.get_or_create_concept("bank")
    assert base_node is not None
    assert base_node.name == "bank"

    # Simulate learning the first sense
    # We use the internal method to pass context
    sense1 = agent.knowledge_manager._get_or_create_concept_node(
        "bank", context_sentence="I went to the bank to deposit money."
    )

    assert sense1 is not None
    assert sense1.name == "bank (financial institution)"

    # Verify edge from base to sense
    edges = agent.graph.get_edges_by_type(base_node.id, "has_sense")
    assert len(edges) == 1
    assert edges[0].target == sense1.id

    # 2. Teach "A bank is the side of a river"
    # Mock disambiguation for the second case
    agent.interpreter.disambiguate_word.return_value = "river edge"

    sense2 = agent.knowledge_manager._get_or_create_concept_node(
        "bank", context_sentence="We sat on the bank of the river."
    )

    assert sense2 is not None
    assert sense2.name == "bank (river edge)"

    # Verify both senses exist
    edges = agent.graph.get_edges_by_type(base_node.id, "has_sense")
    assert len(edges) == 2
    targets = [agent.graph.get_node_by_id(e.target).name for e in edges]
    assert "bank (financial institution)" in targets
    assert "bank (river edge)" in targets


def test_ambiguity_query(agent):
    """Test that querying an ambiguous word returns a clarification request."""

    # Setup the graph with ambiguity
    base_node = agent.knowledge_manager.get_or_create_concept("crane")

    # Manually create sense nodes to bypass _clean_phrase stripping parens
    sense1 = ConceptNode("crane (bird)")
    agent.graph.add_node(sense1)

    sense2 = ConceptNode("crane (machine)")
    agent.graph.add_node(sense2)

    agent.graph.add_edge(base_node, sense1, "has_sense")
    agent.graph.add_edge(base_node, sense2, "has_sense")

    # Mock the conversation manager's fact gathering to return nothing for the base node
    # (since we want to test the ambiguity check, which happens before fact synthesis)
    agent.conversation_manager._gather_facts_multihop = MagicMock(return_value=[])

    # Mock synthesize_answer to just return the input facts (not needed for this test but good practice)
    agent.interpreter.synthesize_answer.return_value = "Mock answer"

    # Query
    response = agent._answer_question_about("crane")

    # Verify response asks for clarification
    assert "can mean multiple things" in response
    assert "Bird" in response
    assert "Machine" in response
