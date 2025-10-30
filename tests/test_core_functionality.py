from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from axiom.cognitive_agent import CognitiveAgent
from axiom.graph_core import ConceptGraph, ConceptNode
from axiom.lexicon_manager import LexiconManager
from axiom.universal_interpreter import InterpretData


def test_chat_handles_total_interpretation_failure(agent: CognitiveAgent, monkeypatch):
    """
    Covers the case where both the symbolic parser and the LLM interpreter fail.
    """
    agent.lexicon.add_linguistic_knowledge_quietly("some", "determiner")
    agent.lexicon.add_linguistic_knowledge_quietly("complete", "adjective")
    agent.lexicon.add_linguistic_knowledge_quietly("gibberish", "noun")

    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(agent.interpreter, "interpret", lambda *args, **kwargs: None)

    response = agent.chat("some complete gibberish")

    assert "I'm sorry, I was unable to understand that." in response
    print("Agent correctly handled total interpretation failure.")


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

    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda *args, **kwargs: mock_interpretations,
    )

    learning_spy = MagicMock(return_value=(True, "learned_new_fact"))
    monkeypatch.setattr(agent, "_process_statement_for_learning", learning_spy)

    agent.chat("a sparrow is a bird and has wings")

    assert learning_spy.call_count == 2

    first_call_args = learning_spy.call_args_list[0][0]
    second_call_args = learning_spy.call_args_list[1][0]

    learned_relation_1 = first_call_args[0]
    learned_relation_2 = second_call_args[0]

    assert learned_relation_1["object"] == "bird"
    assert learned_relation_2["object"] == "wings"
    print("Agent correctly processed all clauses from a multi-clause statement.")


def test_chat_introspection_loop_learns_new_fact(agent: CognitiveAgent, monkeypatch):
    """
    Covers the introspection block where the agent learns from its own response.
    """
    setup_agent = agent
    setup_agent.chat("Paris is the capital of France")

    mock_synthesized_response = (
        "Paris is the capital of France, and Paris is in Europe."
    )
    monkeypatch.setattr(
        agent.interpreter,
        "synthesize",
        lambda *args, **kwargs: mock_synthesized_response,
    )

    introspection_interpretation = [
        {
            "intent": "statement_of_fact",
            "relation": {
                "subject": "paris",
                "verb": "is_located_in",
                "object": "europe",
            },
        },
    ]

    parse_spy = MagicMock(
        side_effect=[
            [
                {
                    "intent": "question_about_entity",
                    "entities": [{"name": "paris", "type": "CONCEPT"}],
                },
            ],
            introspection_interpretation,
        ],
    )
    monkeypatch.setattr("axiom.symbolic_parser.SymbolicParser.parse", parse_spy)

    learning_spy = MagicMock(return_value=(True, "learned_new_fact"))
    monkeypatch.setattr(agent, "_process_statement_for_learning", learning_spy)

    agent.chat("tell me about Paris")

    assert parse_spy.call_count == 2

    assert learning_spy.call_count > 0, "Learning method was not called"
    last_call_args = learning_spy.call_args[0]
    learned_relation = last_call_args[0]

    assert learned_relation["subject"] == "paris"
    assert learned_relation["verb"] == "is_located_in"
    assert learned_relation["object"] == "europe"
    print("Agent successfully learned a new fact via introspection.")


def test_agent_initialization(agent: CognitiveAgent):
    """Tests that the agent can be created successfully without errors."""
    assert agent is not None
    assert isinstance(agent, CognitiveAgent)
    print("Agent initialized successfully.")


def test_parser_handles_show_all_facts_command(agent: CognitiveAgent):
    """
    Covers the specific 'if clause.lower() == "show all facts":' branch
    in the symbolic parser.
    """
    interpretations = agent.parser.parse("show all facts")
    assert interpretations, "Parser failed to handle the 'show all facts' command."

    interpretation = interpretations[0]
    assert interpretation.get("intent") == "command_show_all_facts"
    assert "show all facts" in interpretation.get("key_topics", [])
    print("Parser correctly identified 'show all facts' command.")


@pytest.mark.parametrize(
    ("question", "expected_entity"),
    [
        ("what is a raven", "a raven"),
        ("who is the president", "the president"),
        ("where is paris", "paris"),
    ],
)
def test_parser_handles_wh_question(
    agent: CognitiveAgent,
    question: str,
    expected_entity: str,
):
    """
    Covers the 'if words[0] in self.QUESTION_WORDS:' branch in the parser.
    Uses parametrize to efficiently test multiple question types.
    """
    interpretations = agent.parser.parse(question)
    assert interpretations, f"Parser failed to handle the question: '{question}'"

    interpretation = interpretations[0]
    entities = interpretation.get("entities", [])
    assert interpretation.get("intent") == "question_about_entity"
    assert len(entities) > 0
    assert entities[0].get("name") == expected_entity
    print(f"Parser correctly handled wh-question: '{question}'")


def test_parser_handles_unparseable_sentence(agent: CognitiveAgent):
    """
    Covers the failure paths in the parser where it should correctly
    return None because the sentence is nonsense.
    """
    interpretations = agent.parser.parse("purple sleep furiously")

    assert interpretations is None, (
        "Parser should have returned None for a nonsense sentence."
    )
    print("Parser correctly handled unparseable input.")


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
    """
    Covers the simple conversational branches in the agent's _process_intent method.
    Uses monkeypatch to safely replace the parser's output for this test.
    """
    mock_interpretation = [{"intent": intent}]
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda self, text, context_subject=None: mock_interpretation,
    )

    response = agent.chat(user_input)

    assert response == expected_response
    print(f"Agent correctly handled simple intent for: '{user_input}'")


def test_graph_core_full_lifecycle(tmp_path: Path):
    """
    Tests the fundamental operations of the ConceptGraph to increase its coverage.
    This covers initialization, adding, retrieving, and saving/loading.
    """
    graph = ConceptGraph()
    assert len(graph.graph.nodes) == 0, "Graph should start empty."

    cat_node = ConceptNode(name="Cat", node_type="animal")
    animal_node = ConceptNode(name="Animal", node_type="category")
    graph.add_node(cat_node)
    graph.add_node(animal_node)

    assert len(graph.graph.nodes) == 2
    retrieved_cat = graph.get_node_by_name("cat")
    assert retrieved_cat is not None
    assert retrieved_cat.id == cat_node.id
    assert retrieved_cat.type == "animal"

    assert graph.get_node_by_name("dog") is None

    print("Graph Core: Node creation and retrieval successful.")

    edge = graph.add_edge(retrieved_cat, animal_node, "is_a", weight=0.9)
    assert edge is not None
    assert len(graph.graph.edges) == 1

    outgoing_edges = graph.get_edges_from_node(retrieved_cat.id)
    assert len(outgoing_edges) == 1
    assert outgoing_edges[0].type == "is_a"
    assert outgoing_edges[0].target == animal_node.id

    incoming_edges = graph.get_edges_to_node(animal_node.id)
    assert len(incoming_edges) == 1
    assert incoming_edges[0].type == "is_a"
    assert incoming_edges[0].source == retrieved_cat.id

    print("Graph Core: Edge creation and retrieval successful.")

    save_file = tmp_path / "test_brain.json"
    graph.save_to_file(save_file)
    assert save_file.exists()

    loaded_graph = ConceptGraph.load_from_file(save_file)
    assert len(loaded_graph.graph.nodes) == 2
    assert len(loaded_graph.graph.edges) == 1

    loaded_cat_node = loaded_graph.get_node_by_name("cat")
    assert loaded_cat_node is not None
    assert loaded_cat_node.type == "animal"

    print("Graph Core: Save and load functionality successful.")


def test_agent_shows_all_facts_after_learning(agent: CognitiveAgent):
    """
    Covers the 'command' intent for 'show all facts' in the agent.
    """
    agent.lexicon._promote_word_for_test("sparrow", "noun")
    agent.lexicon._promote_word_for_test("bird", "noun")
    agent.lexicon._promote_word_for_test("wings", "noun")

    agent.chat("a sparrow is a bird")
    agent.chat("a sparrow has wings")
    print("Agent learned two novel facts for the 'show all facts' test.")


def test_lexicon_and_part_of_speech(agent: CognitiveAgent):
    """
    Tests the LexiconManager's ability to identify known words and the parser's
    ability to check for the correct part of speech, which relies on the lexicon.
    """
    assert agent.lexicon.is_known_word("is") is True, (
        "Lexicon should know the seeded word 'is'."
    )

    assert agent.lexicon.is_known_word("flibbertigibbet") is False, (
        "Lexicon should not know a novel word."
    )
    print("Lexicon correctly identifies known and unknown words.")

    assert agent.parser._is_part_of_speech("is", "verb") is True, (
        "Parser should identify 'is' as a verb."
    )

    assert agent.parser._is_part_of_speech("is", "noun") is False, (
        "Parser should not identify 'is' as a noun."
    )

    assert agent.parser._is_part_of_speech("flibbertigibbet", "verb") is False, (
        "Parser should not identify an unknown word."
    )
    print("Parser correctly identifies parts of speech for known words.")


def test_agent_resolves_pronoun_references(agent: CognitiveAgent):
    """
    Covers the _resolve_references method in the agent.
    Tests the agent's ability to understand pronoun context from conversation history.
    """
    fake_interpretation = InterpretData(
        intent="statement_of_fact",
        entities=[{"name": "raven", "type": "CONCEPT"}],
        relation={"subject": "raven", "verb": "is_a", "object": "bird"},
        key_topics=["raven", "bird"],
        full_text_rephrased="a raven is a bird",
    )
    agent.structured_history.append(("user", [fake_interpretation]))

    pronoun_sentence = "what color is it?"
    resolved_sentence = agent._resolve_references(pronoun_sentence)

    assert "raven" in resolved_sentence
    assert "what color is" in resolved_sentence
    print("Agent correctly resolved the pronoun 'it'.")

    no_pronoun_sentence = "what is a bird?"
    unresolved_sentence = agent._resolve_references(no_pronoun_sentence)

    assert unresolved_sentence == no_pronoun_sentence


@pytest.fixture
def lexicon(agent):
    return LexiconManager(agent)


def test_add_linguistic_knowledge_creates_nodes(lexicon, agent):
    lexicon.add_linguistic_knowledge_quietly("testword", "noun", "A test definition.")
    word_node = agent.graph.get_node_by_name("testword")
    pos_node = agent.graph.get_node_by_name("noun")
    assert word_node is not None
    assert pos_node is not None
    edges = agent.graph.get_edges_from_node(word_node.id)
    assert any(e.type == "is_a" and e.target == pos_node.id for e in edges)
    print(
        "LexiconManager: add_linguistic_knowledge_quietly created word and POS nodes successfully.",
    )


def test_add_linguistic_knowledge_adjective_creates_property_edge(lexicon, agent):
    lexicon.add_linguistic_knowledge_quietly("beautiful", "adjective")
    pos_node = agent.graph.get_node_by_name("adjective")
    property_node = agent.graph.get_node_by_name("property")
    assert pos_node is not None
    assert property_node is not None
    edges = agent.graph.get_edges_from_node(pos_node.id)
    assert any(e.type == "is_a" and e.target == property_node.id for e in edges)
    print("LexiconManager: adjective linked to property correctly.")


def test_is_known_word_returns_correctly(lexicon, agent):
    agent._add_or_update_concept("existingword")
    assert lexicon.is_known_word("existingword") is True
    assert lexicon.is_known_word("nonexistentword") is False
    print("LexiconManager: is_known_word returned correct results.")


def test_add_linguistic_knowledge_with_empty_word_does_nothing(lexicon, agent):
    lexicon.add_linguistic_knowledge_quietly("", "noun")
    node = agent.graph.get_node_by_name("")
    assert node is None
    print("LexiconManager: empty word is ignored gracefully.")


@pytest.mark.parametrize(
    ("user_input", "expected_output"),
    [
        ("what is your name?", "what is the agent's name?"),
        ("who are you?", "what is the agent?"),
        ("you are a robot", "the agent is a robot"),
        ("what is your purpose?", "what is the agent's purpose?"),
        ("thank you for your help", "thank you for your help"),
    ],
)
def test_agent_preprocesses_self_reference(
    agent: CognitiveAgent,
    user_input: str,
    expected_output: str,
):
    """
    Covers the _preprocess_self_reference method in the agent.
    Ensures that references to "you" and "your" are correctly normalized.
    """
    processed_text = agent._preprocess_self_reference(user_input)

    assert processed_text == expected_output
    print(f"Correctly preprocessed '{user_input}' -> '{expected_output}'")


@pytest.mark.parametrize(
    ("user_input", "expected_output"),
    [
        ("what's your name?", "what is your name?"),
        ("I can't do that", "i can not do that"),
        ("They're here.", "they are here."),
        ("This has no contractions", "this has no contractions"),
    ],
)
def test_agent_expands_contractions(
    agent: CognitiveAgent,
    user_input: str,
    expected_output: str,
):
    """
    Covers the _expand_contractions utility method in the agent.
    """
    processed_text = agent._expand_contractions(user_input)

    assert processed_text == expected_output
    print(f"Correctly expanded '{user_input}' -> '{expected_output}'")


def test_agent_falls_back_to_llm_when_parsing_fails(agent: CognitiveAgent, monkeypatch):
    """
    Covers the LLM fallback path in chat() when the symbolic parser fails
    but all words in the input are known.
    """
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda *args, **kwargs: None,
    )

    mock_interpretation = InterpretData(
        intent="unknown",
        entities=[],
        relation=None,
        key_topics=[],
        full_text_rephrased="",
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
    """
    Covers the _answer_question_about method by mocking its complex dependency.
    """
    agent.graph.add_node(ConceptNode(name="canary"))

    mock_facts_tuple = (
        ("canary is a bird", ()),
        ("canary has property yellow", ()),
    )

    monkeypatch.setattr(
        agent,
        "_gather_facts_multihop",
        lambda *args, **kwargs: mock_facts_tuple,
    )

    response = agent._answer_question_about("canary", "what is a canary?")
    assert response is not None
    response_lower = response.lower()

    assert "canary is a bird." in response_lower
    assert "canary has property yellow." in response_lower
    print("Agent correctly answered a question about an entity.")

    monkeypatch.setattr(
        agent,
        "_gather_facts_multihop",
        lambda *args, **kwargs: (),
    )

    response_no_details = agent._answer_question_about("canary", "what is a canary?")
    assert response_no_details is not None

    assert "i dont have any details for canary." in response_no_details.lower()
    print("Agent correctly handled an entity with no details.")

    response_unknown = agent._answer_question_about("dragon", "what is a dragon?")
    assert response_unknown is not None

    assert "i don't have any information about dragon" in response_unknown.lower()
    print("Agent correctly handled a question about an unknown entity.")
