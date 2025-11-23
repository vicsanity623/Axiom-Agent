from unittest.mock import MagicMock

import pytest

from axiom.graph_core import ConceptGraph, ConceptNode
from axiom.knowledge_manager import KnowledgeManager


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.graph = ConceptGraph()
    agent._clean_phrase = lambda x: x.lower().strip()
    agent.lexicon.is_known_word.return_value = True
    return agent


def test_km_initialization(mock_agent):
    km = KnowledgeManager(mock_agent)
    assert km.agent == mock_agent


def test_get_or_create_concept_new(mock_agent):
    km = KnowledgeManager(mock_agent)
    # Mock get_word_info_from_wordnet to avoid network calls or complex logic
    # But get_or_create_concept calls it. We might need to mock it in the module.

    node = km.get_or_create_concept("New Concept")
    assert node is not None
    assert node.name == "new concept"
    assert mock_agent.graph.get_node_by_name("new concept") is not None


def test_get_or_create_concept_existing(mock_agent):
    km = KnowledgeManager(mock_agent)
    existing = ConceptNode("existing")
    mock_agent.graph.add_node(existing)

    node = km.get_or_create_concept("Existing")
    assert node.id == existing.id


def test_manual_add_knowledge_quietly(mock_agent):
    km = KnowledgeManager(mock_agent)
    km.manual_add_knowledge_quietly("A", "concept", "is_related_to", "B")

    node_a = mock_agent.graph.get_node_by_name("a")
    node_b = mock_agent.graph.get_node_by_name("b")
    assert node_a is not None
    assert node_b is not None

    edges = mock_agent.graph.get_edges_from_node(node_a.id)
    assert len(edges) == 1
    assert edges[0].target == node_b.id
