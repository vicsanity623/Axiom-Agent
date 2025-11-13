from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from axiom.cognitive_agent import (
    CognitiveAgent,
    InterpretData,
    PropertyData,
    RelationData,
)
from axiom.universal_interpreter import Task

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def agent(tmp_path: Path) -> CognitiveAgent:
    """Provides a CognitiveAgent with expensive dependencies mocked."""
    brain_file = tmp_path / "test_brain.json"
    state_file = tmp_path / "test_state.json"

    with (
        patch("axiom.cognitive_agent.UniversalInterpreter"),
        patch("axiom.cognitive_agent.KnowledgeHarvester"),
    ):
        return CognitiveAgent(
            brain_file=brain_file, state_file=state_file, inference_mode=False
        )


# --- __init__ Tests ---


def test_init_loads_from_file_and_reseeds_on_failure(tmp_path: Path):
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "state.json"
    invalid_brain_data = {
        "nodes": [{"id": "agent_id_123", "name": "agent", "type": "core_identity"}]
    }
    brain_file.write_text(json.dumps(invalid_brain_data))
    state_file.touch()

    with (
        patch("axiom.cognitive_agent.seed_domain_knowledge") as mock_seed_domain,
        patch("axiom.cognitive_agent.seed_core_vocabulary") as mock_seed_vocab,
        patch("axiom.cognitive_agent.CognitiveAgent.save_brain"),
        patch("axiom.cognitive_agent.CognitiveAgent.save_state"),
        patch("axiom.cognitive_agent.UniversalInterpreter"),
    ):
        CognitiveAgent(brain_file=brain_file, state_file=state_file)
        mock_seed_domain.assert_called_once()
        mock_seed_vocab.assert_called_once()


def test_init_raises_error_without_file_or_data(tmp_path: Path):
    with (
        pytest.raises(ValueError, match="Agent must be initialized"),
        patch("axiom.cognitive_agent.UniversalInterpreter"),
    ):
        CognitiveAgent(
            brain_file=tmp_path / "b.json",
            state_file=tmp_path / "s.json",
            load_from_file=False,
        )


# --- chat() and sub-method Tests ---


def test_chat_pipeline_full_flow(agent: CognitiveAgent, monkeypatch):
    # --- FIX (typeddict-item): Create a complete mock object ---
    mock_interp = InterpretData(
        intent="statement_of_fact",
        relation={"subject": "a", "verb": "is", "object": "b"},
        entities=[],
        key_topics=[],
        full_text_rephrased="",
    )
    monkeypatch.setattr(
        agent.conversation_manager,
        "_get_interpretation",
        lambda *args, **kwargs: [mock_interp],
    )
    monkeypatch.setattr(
        agent.conversation_manager, "_update_history", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        agent.conversation_manager,
        "_process_intent",
        lambda *args, **kwargs: "Processed.",
    )
    monkeypatch.setattr(
        agent.conversation_manager,
        "_synthesize_response",
        lambda *args, **kwargs: ("Synthesized.", False),
    )
    response = agent.chat("a is b")
    assert response == "Synthesized."


def test_chat_handles_clarification(agent: CognitiveAgent, monkeypatch):
    agent.is_awaiting_clarification = True
    monkeypatch.setattr(
        agent.conversation_manager,
        "_handle_clarification",
        lambda text: "Clarification received.",
    )
    response = agent.chat("this one")
    assert response == "Clarification received."


def test_chat_handles_total_failure(agent: CognitiveAgent, monkeypatch):
    monkeypatch.setattr(
        agent.conversation_manager, "_get_interpretation", lambda *args: None
    )
    monkeypatch.setattr(
        agent.conversation_manager, "_handle_cognitive_reflex", lambda *args: None
    )
    response = agent.chat("... gibberish ...")
    assert "I'm sorry, I was unable to understand that." in response


def test_get_interpretation_symbolic_success(agent: CognitiveAgent, monkeypatch):
    # --- FIX (typeddict-item): Create a complete mock object ---
    mock_interp = InterpretData(
        intent="unknown",
        entities=[],
        relation=None,
        key_topics=[],
        full_text_rephrased="",
    )
    monkeypatch.setattr(type(agent.parser), "parse", lambda self, text: [mock_interp])
    result = agent.conversation_manager._get_interpretation("test input")
    assert result is not None
    assert result[0]["intent"] == "unknown"


def test_handle_cognitive_reflex_no_unknown_words(agent: CognitiveAgent, monkeypatch):
    monkeypatch.setattr(type(agent.lexicon), "is_known_word", lambda self, word: True)
    response = agent.conversation_manager._handle_cognitive_reflex("all known words")
    assert response is None


def test_handle_cognitive_reflex_queues_high_priority_task(agent: CognitiveAgent):
    input_text = "this is an unknown word"
    agent.high_priority_tasks.clear()
    response = agent.conversation_manager._handle_cognitive_reflex(input_text)
    # --- FIX (operator): Add a guard clause ---
    assert response is not None
    assert "I'm not familiar with the word 'unknown'" in response
    assert len(agent.high_priority_tasks) == 1
    assert agent.high_priority_tasks[0] == Task(task="INVESTIGATE", topic="unknown")


def test_learn_from_introspection_no_context(agent: CognitiveAgent, monkeypatch):
    mock_parse = MagicMock()
    monkeypatch.setattr(type(agent.parser), "parse", mock_parse)
    # --- FIX (typeddict-item): Create a complete mock object ---
    mock_interp = InterpretData(
        intent="greeting",
        entities=[],
        relation=None,
        key_topics=[],
        full_text_rephrased="",
    )
    agent.conversation_manager._learn_from_introspection("response", mock_interp)
    mock_parse.assert_not_called()


def test_resolve_references_no_pronouns(agent: CognitiveAgent):
    text = "the sky is blue"
    assert agent.conversation_manager._resolve_references(text) == text


def test_resolve_references_no_antecedent(agent: CognitiveAgent):
    agent.structured_history = []
    text = "what is it?"
    assert agent.conversation_manager._resolve_references(text) == text


def test_handle_clarification_full_flow(agent: CognitiveAgent, monkeypatch):
    agent.clarification_context = {"subject": "raven", "conflicting_relation": "is_a"}
    # --- FIX (typeddict-item): Create a complete mock object ---
    mock_interp = InterpretData(
        intent="unknown",
        entities=[{"name": "bird", "type": "CONCEPT"}],
        relation=None,
        key_topics=[],
        full_text_rephrased="",
    )
    monkeypatch.setattr(
        agent.interpreter, "interpret", lambda *args, **kwargs: mock_interp
    )
    raven_node = agent.knowledge_manager._add_or_update_concept("raven")
    mammal_node = agent.knowledge_manager._add_or_update_concept("mammal")
    # --- FIX (arg-type): Add guard clauses ---
    assert raven_node is not None
    assert mammal_node is not None
    agent.graph.add_edge(raven_node, mammal_node, "is_a", weight=0.5)
    response = agent.conversation_manager._handle_clarification("it is a bird")
    assert "Thank you for the clarification" in response


def test_get_corrected_entity_no_match(agent: CognitiveAgent):
    agent.knowledge_manager._add_or_update_concept("testing")
    assert agent.conversation_manager._get_corrected_entity("tsting") == "testing"
    assert agent.conversation_manager._get_corrected_entity("xyz") == "xyz"


def test_process_intent_unhandled_intent(agent: CognitiveAgent):
    # --- FIX (typeddict-item): Create a complete mock object ---
    interp = InterpretData(
        intent="unknown",
        entities=[],
        relation=None,
        key_topics=[],
        full_text_rephrased="",
    )
    response = agent.conversation_manager._process_intent(interp, "some input")
    assert "not sure how to process that" in response


def test_find_specific_fact_multiple_results(agent: CognitiveAgent):
    agent_node = agent.knowledge_manager._add_or_update_concept("agent")
    ability1 = agent.knowledge_manager._add_or_update_concept("learn")
    ability2 = agent.knowledge_manager._add_or_update_concept("reason")
    # --- FIX (arg-type): Add guard clauses ---
    assert agent_node is not None
    assert ability1 is not None
    assert ability2 is not None
    agent.graph.add_edge(agent_node, ability1, "has_ability")
    agent.graph.add_edge(agent_node, ability2, "has_ability")
    response = agent.conversation_manager._find_specific_fact("agent", "has_ability")
    assert response is not None
    assert "My abilities include:" in response


def test_answer_question_about_agent_name(agent: CognitiveAgent):
    agent_node = agent.knowledge_manager._add_or_update_concept("agent")
    name_node = agent.knowledge_manager._add_or_update_concept("Axiom")
    # --- FIX (arg-type): Add guard clauses ---
    assert agent_node is not None
    assert name_node is not None
    agent.graph.add_edge(agent_node, name_node, "has_name")
    response = agent.conversation_manager._answer_question_about(
        "agent", "what is your name?"
    )
    assert "My name is Axiom" in response


def test_synthesize_response_non_trigger(agent: CognitiveAgent):
    response, was_used = agent.conversation_manager._synthesize_response(
        "Hello User.", "hello"
    )
    assert response == "Hello User."
    assert not was_used


def test_perform_multi_hop_query_max_hops(agent: CognitiveAgent):
    n1 = agent.knowledge_manager._add_or_update_concept("n1")
    n2 = agent.knowledge_manager._add_or_update_concept("n2")
    n3 = agent.knowledge_manager._add_or_update_concept("n3")
    n4 = agent.knowledge_manager._add_or_update_concept("n4")

    # --- FIX: Break down the compound assertion into atomic assertions ---
    # This is clearer, easier to debug, and provides type guarding for mypy.
    assert n1 is not None
    assert n2 is not None
    assert n3 is not None
    assert n4 is not None

    agent.graph.add_edge(n1, n2, "rel")
    agent.graph.add_edge(n2, n3, "rel")
    agent.graph.add_edge(n3, n4, "rel")

    path = agent.conversation_manager._perform_multi_hop_query(n1, n4, max_hops=2)
    assert path is None


def test_format_path_as_sentence_empty_path(agent: CognitiveAgent):
    assert agent.conversation_manager._format_path_as_sentence([]) == ""


def test_gather_facts_multihop_filtering(agent: CognitiveAgent):
    start_node = agent.knowledge_manager._add_or_update_concept("start")
    assert start_node is not None
    for i in range(12):
        other_node = agent.knowledge_manager._add_or_update_concept(f"node{i}")
        assert other_node is not None

        # --- FIX: Cast the properties dictionary to satisfy mypy ---
        # We are setting an informal property for testing purposes that is not
        # part of the strict PropertyData TypedDict. Casting tells mypy this is intentional.
        test_properties = cast("PropertyData", {"access_count": i})

        agent.graph.add_edge(start_node, other_node, "rel", properties=test_properties)

    facts = agent.conversation_manager._gather_facts_multihop(start_node.id, max_hops=1)
    assert len(facts) == 10


def test_filter_facts_for_temporal_query(agent: CognitiveAgent):
    facts_tuple = (
        (
            "fact1",
            (("effective_date", (datetime.utcnow() - timedelta(days=1)).isoformat()),),
        ),
    )
    filtered = agent.conversation_manager._filter_facts_for_temporal_query(facts_tuple)
    assert "fact1" in filtered


def test_sanitize_sentence_for_learning(agent: CognitiveAgent):
    assert (
        agent.knowledge_manager._sanitize_sentence_for_learning("Text (with parens).")
        == "Text ."
    )


def test_process_statement_for_learning_incomplete_relation(agent: CognitiveAgent):
    relation = cast("RelationData", {"subject": "a", "verb": "is"})
    learned, msg = agent.knowledge_manager._process_statement_for_learning(relation)
    assert not learned
    assert "Incomplete fact structure" in msg


def test_resolve_exclusive_conflict_stalemate(agent: CognitiveAgent):
    sub_node = agent.knowledge_manager._add_or_update_concept("earth")
    obj_node = agent.knowledge_manager._add_or_update_concept("sphere")
    existing_obj = agent.knowledge_manager._add_or_update_concept("oblate spheroid")
    # --- FIX (arg-type): Add guard clauses ---
    assert sub_node is not None
    assert obj_node is not None
    assert existing_obj is not None
    agent.graph.add_edge(
        sub_node, existing_obj, "is_a", weight=0.95, properties={"provenance": "user"}
    )
    relation_data = RelationData(properties={"confidence": 0.95, "provenance": "user"})
    learned, msg = agent.knowledge_manager._resolve_exclusive_conflict(
        sub_node, obj_node, "is_a", relation_data
    )
    assert not learned


def test_add_or_update_concept_quietly_empty_name(agent: CognitiveAgent):
    assert agent.knowledge_manager._add_or_update_concept_quietly("  ") is None


def test_manual_add_knowledge_quietly(agent: CognitiveAgent):
    agent.knowledge_manager.manual_add_knowledge_quietly(
        "cat", "noun", "is_a", "animal"
    )
    assert agent.graph.get_node_by_name("cat") is not None


def test_learn_new_fact_autonomously_success(agent: CognitiveAgent, monkeypatch):
    mock_relations = [RelationData(subject="cat", verb="is_a", object="mammal")]
    monkeypatch.setattr(
        agent.interpreter,
        "decompose_sentence_to_relations",
        lambda text, main_topic: mock_relations,
    )
    monkeypatch.setattr(
        agent.knowledge_manager,
        "_process_statement_for_learning",
        lambda relation, **kwargs: (True, "success"),
    )
    result = agent.knowledge_manager.learn_new_fact_autonomously(
        "A cat is a mammal", "animals"
    )
    assert result is True


def test_learn_new_fact_autonomously_no_relations(agent: CognitiveAgent, monkeypatch):
    monkeypatch.setattr(
        agent.interpreter,
        "decompose_sentence_to_relations",
        lambda text, main_topic: [],
    )
    result = agent.knowledge_manager.learn_new_fact_autonomously(
        "gibberish text", "topic"
    )
    assert result is False


def test_learn_new_fact_autonomously_processing_fails(
    agent: CognitiveAgent, monkeypatch
):
    mock_relations = [RelationData(subject="invalid", verb="relation", object="data")]
    monkeypatch.setattr(
        agent.interpreter,
        "decompose_sentence_to_relations",
        lambda text, main_topic: mock_relations,
    )
    monkeypatch.setattr(
        agent.knowledge_manager,
        "_process_statement_for_learning",
        lambda relation, **kwargs: (False, "failed"),
    )
    result = agent.knowledge_manager.learn_new_fact_autonomously(
        "Invalid fact", "topic"
    )
    assert result is False
