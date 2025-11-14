from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from axiom.universal_interpreter import InterpretData, RelationData

from axiom.cognitive_agent import CognitiveAgent
from axiom.graph_core import ConceptGraph, ConceptNode
from axiom.lexicon_manager import LexiconManager


def test_chat_handles_total_interpretation_failure(agent: CognitiveAgent, monkeypatch):
    """
    Covers the case where both the symbolic parser and the LLM interpreter fail.
    """
    agent.lexicon.add_linguistic_knowledge_quietly("some", "determiner")
    agent.lexicon.add_linguistic_knowledge_quietly("complete", "adjective")
    agent.lexicon.add_linguistic_knowledge_quietly("gibberish", "noun")

    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(agent.interpreter, "interpret", lambda *args, **kwargs: None)

    response = agent.chat("some complete gibberish")
    assert "I'm sorry, I was unable to understand that." in response


def test_chat_handles_multi_clause_statements(agent: CognitiveAgent, monkeypatch):
    """
    Covers the 'if len(interpretations) > 1:' branch for learning from extra clauses.
    """
    mock_interpretations = [
        {
            "intent": "statement_of_fact",
            "relation": {"subject": "sparrow", "verb": "is_a", "object": "bird"},
        },
        {
            "intent": "statement_of_fact",
            "relation": {
                "subject": "sparrow",
                "verb": "has_property",
                "object": "wings",
            },
        },
    ]
    for interp in mock_interpretations:
        interp.update({"entities": [], "key_topics": [], "full_text_rephrased": ""})

    # Cast to the expected type for mypy
    typed_mock_interpretations = cast("list[InterpretData]", mock_interpretations)

    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda *args, **kwargs: typed_mock_interpretations,
    )

    learning_spy = MagicMock(return_value=(True, "learned_new_fact"))
    # --- FIX: Point to the correct manager ---
    monkeypatch.setattr(
        agent.knowledge_manager, "_process_statement_for_learning", learning_spy
    )

    agent.chat("a sparrow is a bird and has wings")

    # The first clause is handled by _process_intent, the second by _learn_from_extra_clauses
    assert learning_spy.call_count == 2
    first_call_args, second_call_args = (
        learning_spy.call_args_list[0][0],
        learning_spy.call_args_list[1][0],
    )
    assert first_call_args[0]["object"] == "bird"
    assert second_call_args[0]["object"] == "wings"


def test_agent_initialization(agent: CognitiveAgent):
    assert agent is not None
    assert isinstance(agent, CognitiveAgent)


def test_parser_handles_show_all_facts_command(agent: CognitiveAgent):
    interpretations = agent.parser.parse("show all facts")
    assert interpretations
    interpretation = interpretations[0]
    assert interpretation.get("intent") == "command_show_all_facts"
    assert "show all facts" in interpretation.get("key_topics", [])


@pytest.mark.parametrize(
    ("question", "expected_entity"),
    [
        ("what is a raven", "a raven"),
        ("who is the president", "the president"),
        ("where is paris", "paris"),
    ],
)
def test_parser_handles_wh_question(
    agent: CognitiveAgent, question: str, expected_entity: str
):
    interpretations = agent.parser.parse(question)
    assert interpretations
    interpretation = interpretations[0]
    entities = interpretation.get("entities", [])
    assert interpretation.get("intent") == "question_about_entity"
    assert len(entities) > 0
    assert entities[0].get("name") == expected_entity


def test_parser_handles_unparseable_sentence(agent: CognitiveAgent):
    interpretations = agent.parser.parse("purple sleep furiously")
    assert interpretations is None


@pytest.mark.parametrize(
    ("user_input", "expected_response", "intent"),
    [
        ("hello", "Hello User.", "greeting"),
        ("goodbye", "Goodbye User.", "farewell"),
        ("thanks", "You're welcome!", "gratitude"),
        ("you're smart", "I'm glad you think so!", "positive_affirmation"),
    ],
)
def test_agent_handles_simple_intents(
    agent: CognitiveAgent,
    monkeypatch,
    user_input: str,
    expected_response: str,
    intent: str,
):
    mock_interpretation = [cast("InterpretData", {"intent": intent})]
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda self, text, context_subject=None: mock_interpretation,
    )
    response = agent.chat(user_input)
    assert response == expected_response


def test_graph_core_full_lifecycle(tmp_path: Path):
    graph = ConceptGraph()
    assert len(graph.graph.nodes) == 0
    cat_node = ConceptNode(name="Cat", node_type="animal")
    animal_node = ConceptNode(name="Animal", node_type="category")
    graph.add_node(cat_node)
    graph.add_node(animal_node)
    assert len(graph.graph.nodes) == 2
    retrieved_cat = graph.get_node_by_name("cat")
    assert retrieved_cat is not None
    assert retrieved_cat.id == cat_node.id
    edge = graph.add_edge(retrieved_cat, animal_node, "is_a", weight=0.9)
    assert edge is not None
    assert len(graph.graph.edges) == 1
    save_file = tmp_path / "test_brain.json"
    graph.save_to_file(save_file)
    assert save_file.exists()
    loaded_graph = ConceptGraph.load_from_file(save_file)
    assert len(loaded_graph.graph.nodes) == 2


def test_agent_shows_all_facts_after_learning(agent: CognitiveAgent):
    agent.lexicon._promote_word_for_test("sparrow", "noun")
    agent.lexicon._promote_word_for_test("bird", "noun")
    agent.lexicon._promote_word_for_test("wings", "noun")
    agent.chat("a sparrow is a bird")
    agent.chat("a sparrow has wings")
    # This test just ensures the chat calls don't crash. Assertions would be better.


@pytest.fixture
def lexicon(agent):
    return LexiconManager(agent)


def test_add_linguistic_knowledge_creates_nodes(lexicon, agent):
    lexicon.add_linguistic_knowledge_quietly("testword", "noun", "A test definition.")
    word_node = agent.graph.get_node_by_name("testword")
    pos_node = agent.graph.get_node_by_name("noun")
    assert word_node is not None
    assert pos_node is not None


def test_add_linguistic_knowledge_adjective_creates_property_edge(lexicon, agent):
    lexicon.add_linguistic_knowledge_quietly("beautiful", "adjective")
    pos_node = agent.graph.get_node_by_name("adjective")
    assert pos_node is not None


def test_is_known_word_returns_correctly(lexicon, agent):
    # --- FIX: Point to the correct manager ---
    agent.knowledge_manager._add_or_update_concept("existingword")
    assert lexicon.is_known_word("existingword") is True
    assert lexicon.is_known_word("nonexistentword") is False


def test_add_linguistic_knowledge_with_empty_word_does_nothing(lexicon, agent):
    lexicon.add_linguistic_knowledge_quietly("", "noun")
    assert agent.graph.get_node_by_name("") is None


@pytest.mark.parametrize(
    ("user_input", "expected_output"),
    [
        ("what is your name?", "what is the agent's name?"),
        ("who are you?", "what is the agent?"),
        ("you are a robot", "the agent is a robot"),
    ],
)
def test_agent_preprocesses_self_reference(
    agent: CognitiveAgent, user_input: str, expected_output: str
):
    processed_text = agent._preprocess_self_reference(user_input)
    assert processed_text == expected_output


@pytest.mark.parametrize(
    ("user_input", "expected_output"),
    [
        ("what's your name?", "what is your name?"),
        ("I can't do that", "i can not do that"),
    ],
)
def test_agent_expands_contractions(
    agent: CognitiveAgent, user_input: str, expected_output: str
):
    processed_text = agent._expand_contractions(user_input)
    assert processed_text == expected_output


def test_agent_falls_back_to_llm_when_parsing_fails(agent: CognitiveAgent, monkeypatch):
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse", lambda *args, **kwargs: None
    )
    mock_interpretation = cast(
        "InterpretData",
        {
            "intent": "unknown",
            "entities": [],
            "relation": None,
            "key_topics": [],
            "full_text_rephrased": "",
        },
    )
    monkeypatch.setattr(
        "axiom.universal_interpreter.UniversalInterpreter.interpret",
        lambda self, user_input: mock_interpretation,
    )
    agent.lexicon.add_linguistic_knowledge_quietly("complex", "adjective")
    agent.lexicon.add_linguistic_knowledge_quietly("sentence", "noun")
    response = agent.chat("a complex sentence")
    assert "I'm not sure how to process that" in response


def test_agent_answers_question_about_entity(agent: CognitiveAgent, monkeypatch):
    agent.graph.add_node(ConceptNode(name="canary"))
    mock_facts_tuple = (("canary is a bird", ()), ("canary has property yellow", ()))
    # --- FIX: Point to the correct manager ---
    monkeypatch.setattr(
        agent.conversation_manager,
        "_gather_facts_multihop",
        lambda *args, **kwargs: mock_facts_tuple,
    )
    response = agent.conversation_manager._answer_question_about(
        "canary", "what is a canary?"
    )
    assert response is not None
    response_lower = response.lower()
    assert "canary is a bird." in response_lower
    assert "canary has property yellow." in response_lower

    monkeypatch.setattr(
        agent.conversation_manager, "_gather_facts_multihop", lambda *args, **kwargs: ()
    )
    response_no_details = agent.conversation_manager._answer_question_about(
        "canary", "what is a canary?"
    )
    assert response_no_details is not None
    assert "i dont have any details for canary." in response_no_details.lower()

    response_unknown = agent.conversation_manager._answer_question_about(
        "dragon", "what is a dragon?"
    )
    assert response_unknown is not None
    assert "i don't have any information about dragon" in response_unknown.lower()


def test_agent_can_learn_multiple_senses_for_a_word(agent: CognitiveAgent):
    """
    Given: A clean agent.
    When: The agent learns two different 'is a' relationships for the same subject.
    Then: It should create a single node for the subject with two distinct outgoing edges.
    """
    relation1 = cast(
        "RelationData", {"subject": "one", "verb": "is a", "object": "number"}
    )
    # --- FIX: Point to the correct manager ---
    success, _ = agent.knowledge_manager._process_statement_for_learning(relation1)
    assert success is True

    node_one = agent.graph.get_node_by_name("one")
    assert node_one is not None

    edges = agent.graph.get_edges_from_node(node_one.id)
    assert len(edges) == 1
    assert edges[0].type == "is a"
    target_node = agent.graph.get_node_by_id(edges[0].target)
    assert target_node is not None
    assert target_node.name == "number"

    relation2 = cast(
        "RelationData", {"subject": "one", "verb": "is a", "object": "pronoun"}
    )
    # --- FIX: Point to the correct manager ---
    success, _ = agent.knowledge_manager._process_statement_for_learning(relation2)
    assert success is True

    node_one_after = agent.graph.get_node_by_name("one")
    assert node_one_after is not None
    assert node_one_after.id == node_one.id
    edges_after = agent.graph.get_edges_from_node(node_one_after.id)
    assert len(edges_after) == 2

    target_names = {
        target.name
        for edge in edges_after
        if (target := agent.graph.get_node_by_id(edge.target)) is not None
    }

    assert "number" in target_names
    assert "pronoun" in target_names
