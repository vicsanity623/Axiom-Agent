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
    # GIVEN: All words in the input are known to the agent's lexicon.
    agent.lexicon.add_linguistic_knowledge_quietly("some", "determiner")
    agent.lexicon.add_linguistic_knowledge_quietly("complete", "adjective")
    agent.lexicon.add_linguistic_knowledge_quietly("gibberish", "noun")

    # AND: The symbolic parser is mocked to fail.
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda *args, **kwargs: None,
    )

    # AND: The LLM interpreter is mocked to fail.
    monkeypatch.setattr(agent.interpreter, "interpret", lambda *args, **kwargs: None)

    # WHEN: We call the chat method with an input that is lexically known but structurally unparseable.
    response = agent.chat("some complete gibberish")

    # THEN: The agent should now correctly fall through to its final "I don't understand" message.
    assert "I'm sorry, I was unable to understand that." in response
    print("Agent correctly handled total interpretation failure.")


def test_chat_handles_multi_clause_statements(agent: CognitiveAgent, monkeypatch):
    """
    Covers the 'if len(interpretations) > 1:' branch for learning from extra clauses.
    """
    # GIVEN: A mock parser that returns multiple interpretations (two facts).
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
    # We must use a fully compliant TypedDict structure
    for interp in mock_interpretations:
        interp.update({"entities": [], "key_topics": [], "full_text_rephrased": ""})

    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda *args, **kwargs: mock_interpretations,
    )

    # AND: We create a "spy" to watch the _process_statement_for_learning method.
    learning_spy = MagicMock(return_value=(True, "learned_new_fact"))
    monkeypatch.setattr(agent, "_process_statement_for_learning", learning_spy)

    # WHEN: We call the chat method.
    agent.chat("a sparrow is a bird and has wings")

    # THEN: The learning method should have been called exactly TWICE.
    assert learning_spy.call_count == 2

    # Verify the contents of both calls to be certain.
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
    # GIVEN: The agent knows a fact about Paris.
    # We must use a separate agent instance for setup because monkeypatching the class affects all instances.
    setup_agent = agent
    setup_agent.chat("Paris is the capital of France")

    # AND: We will make the mock synthesizer's response contain a NEW, related fact.
    mock_synthesized_response = (
        "Paris is the capital of France, and Paris is in Europe."
    )
    monkeypatch.setattr(
        agent.interpreter,
        "synthesize",
        lambda *args, **kwargs: mock_synthesized_response,
    )

    # AND: We will make the mock parser recognize the new fact during introspection.
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

    # We need a spy. It will have multiple return values: one for the user's question, one for introspection.
    parse_spy = MagicMock(
        side_effect=[
            # First call (for "tell me about Paris") returns a standard question intent.
            [
                {
                    "intent": "question_about_entity",
                    "entities": [{"name": "paris", "type": "CONCEPT"}],
                },
            ],
            # Second call (for the synthesized response) returns the new fact.
            introspection_interpretation,
        ],
    )
    monkeypatch.setattr("axiom.symbolic_parser.SymbolicParser.parse", parse_spy)

    # AND: We spy on the final learning function.
    learning_spy = MagicMock(return_value=(True, "learned_new_fact"))
    monkeypatch.setattr(agent, "_process_statement_for_learning", learning_spy)

    # WHEN: The user asks the question that triggers the synthesized response.
    agent.chat("tell me about Paris")

    # THEN: Assert that the parser was called twice.
    assert parse_spy.call_count == 2

    # AND: Assert that the learning function was called with the new fact from introspection.
    # The first call to learning is for the setup fact. The second is for introspection.
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


def test_learning_a_fact(agent: CognitiveAgent):
    """Tests that the agent can learn a simple fact and store it in its graph."""
    # 1. Action: Teach the agent a new fact.
    response = agent.chat("a horse is an animal")
    assert "I understand" in response

    # 2. Verification: Check the agent's brain to see if it learned correctly.
    fact_is_known = False
    horse_node = agent.graph.get_node_by_name("horse")
    assert horse_node is not None, "Agent did not create a node for 'horse'."

    for edge in agent.graph.get_edges_from_node(horse_node.id):
        target_node = agent.graph.get_node_by_id(edge.target)
        if edge.type == "is_a" and target_node and target_node.name == "animal":
            fact_is_known = True
            break

    assert fact_is_known is True, "Agent failed to learn 'horse is an animal'."
    print("Agent successfully learned a fact.")


@pytest.mark.parametrize(
    ("sentence", "expected_subject", "expected_relation", "expected_object"),
    [
        ("Paris is a city in France", "paris", "is_located_in", "france"),
        ("a wheel is part of a car", "wheel", "is_part_of", "car"),
    ],
)
def test_parser_handles_prepositions(
    agent: CognitiveAgent,
    sentence,
    expected_subject,
    expected_relation,
    expected_object,
):
    """Tests that the symbolic parser can handle different prepositional phrases."""
    # 1. Action: Parse the sentence
    interpretations = agent.parser.parse(sentence)
    assert interpretations, "Parser failed to produce an interpretation."

    relation = interpretations[0].get("relation")
    assert relation, "Parser failed to extract a relation."

    # 2. Verification: Check if the components are correct
    assert relation.get("subject") == expected_subject
    assert relation.get("verb") == expected_relation
    assert relation.get("object") == expected_object
    print(f"Parser correctly handled: '{sentence}'")


def test_parser_handles_show_all_facts_command(agent: CognitiveAgent):
    """
    Covers the specific 'if clause.lower() == "show all facts":' branch
    in the symbolic parser.
    """
    # 1. Action: Send the specific command to the parser.
    interpretations = agent.parser.parse("show all facts")
    assert interpretations, "Parser failed to handle the 'show all facts' command."

    # 2. Verification: Check for the correct intent.
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
    # 1. Action: Parse the question.
    interpretations = agent.parser.parse(question)
    assert interpretations, f"Parser failed to handle the question: '{question}'"

    # 2. Verification: Check for correct intent and entity extraction.
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
    # 1. Action: Parse a sentence that doesn't match any rules.
    interpretations = agent.parser.parse("purple sleep furiously")

    # 2. Verification: Ensure the parser correctly gives up.
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
    # 1. Action: Tell monkeypatch to temporarily replace the real SymbolicParser.parse method.
    # This is the correct, robust way to mock.
    mock_interpretation = [{"intent": intent}]
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda self, text, context_subject=None: mock_interpretation,
    )

    # Now, when agent.chat calls agent.parser.parse, our fake lambda will run instead.
    response = agent.chat(user_input)

    # 2. Verification: Check for the correct canned response.
    assert response == expected_response
    print(f"Agent correctly handled simple intent for: '{user_input}'")


def test_graph_core_full_lifecycle(tmp_path: Path):
    """
    Tests the fundamental operations of the ConceptGraph to increase its coverage.
    This covers initialization, adding, retrieving, and saving/loading.
    """
    # 1. Initialization and Node Operations
    graph = ConceptGraph()
    assert len(graph.graph.nodes) == 0, "Graph should start empty."

    # Create and add two nodes
    cat_node = ConceptNode(name="Cat", node_type="animal")  # Name will be lowercased
    animal_node = ConceptNode(name="Animal", node_type="category")
    graph.add_node(cat_node)
    graph.add_node(animal_node)

    # Verify nodes were added and can be retrieved
    assert len(graph.graph.nodes) == 2
    retrieved_cat = graph.get_node_by_name("cat")
    assert retrieved_cat is not None
    assert retrieved_cat.id == cat_node.id
    assert retrieved_cat.type == "animal"

    # Test retrieving a non-existent node
    assert graph.get_node_by_name("dog") is None

    print("Graph Core: Node creation and retrieval successful.")

    # 2. Edge Operations
    edge = graph.add_edge(retrieved_cat, animal_node, "is_a", weight=0.9)
    assert edge is not None
    assert len(graph.graph.edges) == 1

    # Verify edge can be retrieved from the source node
    outgoing_edges = graph.get_edges_from_node(retrieved_cat.id)
    assert len(outgoing_edges) == 1
    assert outgoing_edges[0].type == "is_a"
    assert outgoing_edges[0].target == animal_node.id

    # Verify edge can be retrieved from the target node
    incoming_edges = graph.get_edges_to_node(animal_node.id)
    assert len(incoming_edges) == 1
    assert incoming_edges[0].type == "is_a"
    assert incoming_edges[0].source == retrieved_cat.id

    print("Graph Core: Edge creation and retrieval successful.")

    # 3. Save and Load Operations
    # Use the tmp_path fixture provided by pytest for a temporary file
    save_file = tmp_path / "test_brain.json"
    graph.save_to_file(save_file)
    assert save_file.exists()

    # Load the graph into a new instance
    loaded_graph = ConceptGraph.load_from_file(save_file)
    assert len(loaded_graph.graph.nodes) == 2
    assert len(loaded_graph.graph.edges) == 1

    # Verify the content of the loaded graph
    loaded_cat_node = loaded_graph.get_node_by_name("cat")
    assert loaded_cat_node is not None
    assert loaded_cat_node.type == "animal"

    print("Graph Core: Save and load functionality successful.")


def test_agent_answers_yes_no_question(agent: CognitiveAgent, monkeypatch):
    """
    Covers the 'question_yes_no' branch. Uses a monkeypatch context manager
    to ensure the mock does not leak to other tests.
    """
    # 1. Setup: Teach the agent a foundational fact using its real parser.
    agent.chat("a raven is a bird")

    # This helper function now uses a context manager for perfect isolation.
    def mock_and_chat(user_input: str, subject: str, object_: str) -> str:
        with monkeypatch.context() as m:
            mock_relation = {"subject": subject, "verb": "is", "object": object_}
            mock_interpretation = [
                {
                    "intent": "question_yes_no",
                    "relation": mock_relation,
                    "entities": [],
                    "key_topics": [],
                    "full_text_rephrased": "",
                },
            ]

            # The 'm.setattr' call patches the parser.
            m.setattr(
                "axiom.symbolic_parser.SymbolicParser.parse",
                lambda *args, **kwargs: mock_interpretation,
            )
            # The patch is ONLY active inside this 'with' block.
            return agent.chat(user_input)

    # 2. Test the "Yes" case
    response_yes = mock_and_chat("is a raven a bird?", "raven", "bird")
    assert "Yes" in response_yes
    print("Agent correctly answered 'Yes'.")

    # 3. Test the "No" case
    # Teach the agent a conflicting fact.
    agent.chat("a raven is not a mammal")
    response_no = mock_and_chat("is a raven a mammal?", "raven", "mammal")
    assert "No" in response_no
    print("Agent correctly handled a contradictory question.")

    # 4. Test the "I don't know" case
    response_unknown = mock_and_chat("is a raven black?", "raven", "black")
    assert "don't have information" in response_unknown or "No" in response_unknown
    print("Agent correctly answered for an unknown property.")


def test_agent_shows_all_facts_after_learning(agent: CognitiveAgent):
    """
    Covers the 'command' intent for 'show all facts' in the agent.
    Ensures the agent can retrieve and format all known facts from its graph
    using novel facts not present in the seeded knowledge base.
    """
    # 1. Setup: Teach the agent two distinct, novel facts.
    agent.chat("a sparrow is a bird")
    agent.chat("a sparrow has wings")
    print("Agent learned two novel facts for the 'show all facts' test.")

    # 2. Action: Issue the command to the agent.
    response = agent.chat("show all facts")

    # 3. Verification: Check that the response contains the expected introductory
    #    text and the core components of the facts we just taught it.
    assert "Here are all the high-confidence facts I know" in response

    # Lowercase the response for robust, case-insensitive checks.
    response_lower = response.lower()

    # Check for the first learned fact
    assert "sparrow" in response_lower
    assert "is_a" in response_lower
    assert "bird" in response_lower

    # Check for the second learned fact
    # Note: The symbolic parser will likely turn "has" into "has_property"
    assert "has_property" in response_lower
    assert "wing" in response_lower  # Note: lemmatizer turns "wings" into "wing"

    print("Agent correctly listed all newly learned facts in its response.")


def test_lexicon_and_part_of_speech(agent: CognitiveAgent):
    """
    Tests the LexiconManager's ability to identify known words and the parser's
    ability to check for the correct part of speech, which relies on the lexicon.
    """
    # 1. Test is_known_word() from LexiconManager
    # The agent's lexicon is seeded with common words like "is".
    assert agent.lexicon.is_known_word("is") is True, (
        "Lexicon should know the seeded word 'is'."
    )

    # Test a word that is definitely not in the seeded lexicon.
    assert agent.lexicon.is_known_word("flibbertigibbet") is False, (
        "Lexicon should not know a novel word."
    )
    print("Lexicon correctly identifies known and unknown words.")

    # 2. Test _is_part_of_speech() from SymbolicParser (which uses the graph/lexicon)
    # The agent knows "is" is a verb from the knowledge base seeding.
    assert agent.parser._is_part_of_speech("is", "verb") is True, (
        "Parser should identify 'is' as a verb."
    )

    # Test the negative case: "is" is not a noun.
    assert agent.parser._is_part_of_speech("is", "noun") is False, (
        "Parser should not identify 'is' as a noun."
    )

    # Test an unknown word: it can't be any part of speech.
    assert agent.parser._is_part_of_speech("flibbertigibbet", "verb") is False, (
        "Parser should not identify an unknown word."
    )
    print("Parser correctly identifies parts of speech for known words.")


def test_agent_resolves_pronoun_references(agent: CognitiveAgent):
    """
    Covers the _resolve_references method in the agent.
    Tests the agent's ability to understand pronoun context from conversation history.
    """
    # 1. Setup: Manually create a fake conversation history to establish context.
    # We pretend the user just said "a raven is a bird".
    fake_interpretation = InterpretData(
        intent="statement_of_fact",
        entities=[{"name": "raven", "type": "CONCEPT"}],
        relation={"subject": "raven", "verb": "is_a", "object": "bird"},
        key_topics=["raven", "bird"],  # Add a plausible value
        full_text_rephrased="a raven is a bird",  # Add a plausible value
    )
    agent.structured_history.append(("user", [fake_interpretation]))

    # 2. Action & Verification (Positive Case):
    # Ask a question with a pronoun and check if it's replaced correctly.
    pronoun_sentence = "what color is it?"
    resolved_sentence = agent._resolve_references(pronoun_sentence)

    # The agent should replace "it" with the last subject, "raven".
    assert "raven" in resolved_sentence
    assert "what color is" in resolved_sentence
    print("Agent correctly resolved the pronoun 'it'.")

    # 3. Action & Verification (Negative Case):
    # Use a sentence without a pronoun to ensure it's not changed.
    no_pronoun_sentence = "what is a bird?"
    unresolved_sentence = agent._resolve_references(no_pronoun_sentence)

    assert unresolved_sentence == no_pronoun_sentence


def test_agent_performs_multi_hop_query(agent: CognitiveAgent, monkeypatch):
    """
    Covers the multi-hop query logic in the agent.
    Tests if the agent can connect two concepts through a chain of facts.
    """
    # 1. Setup: Teach the agent a chain of facts.
    agent.chat("socrates is a philosopher")
    agent.chat("a philosopher is a person")

    # 2. Mock the Parser for the success case
    mock_relation = {"subject": "socrates", "verb": "is a", "object": "person"}
    mock_interpretation = [
        {"intent": "question_about_entity", "relation": mock_relation},
    ]
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda self, text, context_subject=None: mock_interpretation,
    )

    # 3. Action
    response = agent.chat("what is socrates to a person?")

    # 4. Verification
    response_lower = response.lower()
    assert "based on what i know" in response_lower
    assert "socrates is a philosopher" in response_lower
    assert "which in turn is a person" in response_lower

    # 5. Test Failure Case
    mock_relation_fail = {"subject": "socrates", "verb": "is a", "object": "animal"}
    mock_interpretation_fail = [
        {"intent": "question_about_entity", "relation": mock_relation_fail},
    ]
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda self, text, context_subject=None: mock_interpretation_fail,
    )

    response_fail = agent.chat("what is socrates to an animal?")
    assert "don't know of a direct relationship" in response_fail.lower()


# --- LexiconManager tests ---


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
    # Should not create a node for an empty string
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
        # Test the negative lookbehind to ensure "thank you" is not changed
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
    # 1. Action: Call the preprocessing method directly.
    processed_text = agent._preprocess_self_reference(user_input)

    # 2. Verification: Check that the text was transformed as expected.
    assert processed_text == expected_output
    print(f"Correctly preprocessed '{user_input}' -> '{expected_output}'")


@pytest.mark.parametrize(
    ("user_input", "expected_output"),
    [
        ("what's your name?", "what is your name?"),
        ("I can't do that", "i can not do that"),
        ("They're here.", "they are here."),
        # Test a sentence with no contractions to ensure it's unchanged
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
    # 1. Action: Call the contraction expansion method directly.
    processed_text = agent._expand_contractions(user_input)

    # 2. Verification: Check that the text was transformed as expected.
    # Note: The method also lowercases the text, so the expected output is lowercase.
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

    # "a" will now be known after the bug fix.
    agent.lexicon.add_linguistic_knowledge_quietly("complex", "adjective")
    agent.lexicon.add_linguistic_knowledge_quietly("sentence", "noun")

    response = agent.chat("a complex sentence")

    assert "I'm not sure how to process that" in response


def test_agent_creates_goal_for_unknown_word(agent: CognitiveAgent, monkeypatch):
    """
    Covers the "cognitive reflex" in chat() where an unknown word is detected.
    """
    monkeypatch.setattr(
        "axiom.symbolic_parser.SymbolicParser.parse",
        lambda *args, **kwargs: None,
    )

    response = agent.chat("what is a flibbertigibbet")

    assert "New word 'flibbertigibbet' discovered" in response
    assert "INVESTIGATE: flibbertigibbet" in agent.learning_goals


def test_agent_answers_question_about_entity(agent: CognitiveAgent, monkeypatch):
    """
    Covers the _answer_question_about method by mocking its complex dependency.
    """
    # --- SUCCESS PATH ---

    # 1. GIVEN: The agent's graph MUST contain the node we are asking about.
    # This is the crucial step we were missing.
    agent.graph.add_node(ConceptNode(name="canary"))

    # AND: We define the data our mock will return when fact-gathering is successful.
    mock_facts_tuple = (
        ("canary is a bird", ()),
        ("canary has property yellow", ()),
    )

    # 2. MOCK: Replace the _gather_facts_multihop method to return our data.
    monkeypatch.setattr(
        agent,
        "_gather_facts_multihop",
        lambda *args, **kwargs: mock_facts_tuple,
    )

    # 3. WHEN: Call the method under test.
    response = agent._answer_question_about("canary", "what is a canary?")
    assert response is not None
    response_lower = response.lower()

    # 4. THEN: Verify that _answer_question_about correctly formats the facts.
    assert "canary is a bird." in response_lower
    assert "canary has property yellow." in response_lower
    print("Agent correctly answered a question about an entity.")

    # --- FAILURE PATH (NO DETAILS FOUND) ---

    # 1. GIVEN: The 'canary' node still exists.
    # 2. MOCK: Now, we mock the gatherer to return an EMPTY tuple.
    monkeypatch.setattr(
        agent,
        "_gather_facts_multihop",
        lambda *args, **kwargs: (),
    )

    # 3. WHEN: We call the method again.
    response_no_details = agent._answer_question_about("canary", "what is a canary?")
    assert response_no_details is not None

    # 4. THEN: Assert it returns the "no details" message.
    assert "i dont have any details for canary." in response_no_details.lower()
    print("Agent correctly handled an entity with no details.")

    # --- FAILURE PATH (SUBJECT DOES NOT EXIST) ---

    # 1. GIVEN: The graph does not contain a "dragon" node.
    # 2. WHEN: We ask about a completely unknown entity.
    response_unknown = agent._answer_question_about("dragon", "what is a dragon?")
    assert response_unknown is not None

    # 3. THEN: Assert it returns the "no information" message.
    assert "i don't have any information about dragon" in response_unknown.lower()
    print("Agent correctly handled a question about an unknown entity.")
