from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from axiom.cognitive_agent import (
    CognitiveAgent,
    ConceptNode,
    InterpretData,
    RelationData,
)

if TYPE_CHECKING:
    from pathlib import Path

    from axiom.graph_core import RelationshipEdge


@pytest.fixture
def agent(tmp_path: Path) -> CognitiveAgent:
    """Provides a CognitiveAgent with expensive dependencies mocked."""
    brain_file = tmp_path / "test_brain.json"
    state_file = tmp_path / "test_state.json"

    with (
        patch("axiom.cognitive_agent.UniversalInterpreter") as mockinterpreter,
        patch("axiom.cognitive_agent.KnowledgeHarvester") as mockharvester,
    ):
        mock_interpreter_instance = mockinterpreter.return_value
        mock_interpreter_instance.synthesis_cache = {}
        mock_interpreter_instance.interpretation_cache = {}

        mock_harvester_instance = mockharvester.return_value

        agent = CognitiveAgent(
            brain_file=brain_file, state_file=state_file, inference_mode=False
        )
        agent.interpreter = mock_interpreter_instance
        agent.harvester = mock_harvester_instance

        return agent


# --- __init__ Tests ---


def test_init_loads_from_file_and_reseeds_on_failure(tmp_path: Path):
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "state.json"
    invalid_brain = {"nodes": [{"id": "agent", "name": "agent"}]}
    brain_file.write_text(json.dumps(invalid_brain))

    with (
        patch("axiom.cognitive_agent.seed_domain_knowledge") as mock_seed_domain,
        patch("axiom.cognitive_agent.seed_core_vocabulary") as mock_seed_vocab,
        patch("axiom.cognitive_agent.UniversalInterpreter"),
    ):
        CognitiveAgent(brain_file=brain_file, state_file=state_file)
        mock_seed_domain.assert_called_once()
        mock_seed_vocab.assert_called_once()


def test_init_from_axm_data(tmp_path: Path):
    """Test initialization from pre-loaded dictionary data."""
    brain_data = {
        "nodes": [{"id": "node1", "name": "test", "type": "concept"}],
        "links": [],
        "learning_iterations": 10,  # move here
    }
    cache_data = {"interpretations": [["key1", {}]], "synthesis": [["key2", "value"]]}

    with patch("axiom.cognitive_agent.UniversalInterpreter"):
        agent = CognitiveAgent(
            brain_file=tmp_path / "b.json",
            state_file=tmp_path / "s.json",
            load_from_file=False,
            brain_data=brain_data,
            cache_data=cache_data,
        )

    assert agent.learning_iterations == 10
    assert "key1" in agent.interpreter.interpretation_cache
    assert "key2" in agent.interpreter.synthesis_cache


def test_init_raises_error_without_file_or_data(tmp_path: Path):
    with (
        pytest.raises(
            ValueError, match="Agent must be initialized with either files or data."
        ),
        patch("axiom.cognitive_agent.UniversalInterpreter"),
    ):
        CognitiveAgent(
            brain_file=tmp_path / "b.json",
            state_file=tmp_path / "s.json",
            load_from_file=False,
        )


# --- chat() and sub-method Tests ---


def test_chat_pipeline_full_flow(agent: CognitiveAgent, monkeypatch):
    mock_interp = InterpretData(
        intent="statement_of_fact",
        relation={"subject": "a", "verb": "is", "object": "b"},
        entities=[],
        key_topics=[],
        full_text_rephrased="",
    )
    monkeypatch.setattr(
        agent, "_get_interpretation", lambda *args, **kwargs: [mock_interp]
    )
    monkeypatch.setattr(agent, "_update_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_process_intent", lambda *args, **kwargs: "Processed.")
    monkeypatch.setattr(
        agent, "_synthesize_response", lambda *args, **kwargs: ("Synthesized.", False)
    )

    response = agent.chat("a is b")
    assert response == "Synthesized."


def test_chat_handles_clarification(agent: CognitiveAgent, monkeypatch):
    agent.is_awaiting_clarification = True
    monkeypatch.setattr(
        agent, "_handle_clarification", lambda text: "Clarification received."
    )

    response = agent.chat("this one")
    assert response == "Clarification received."


def test_chat_handles_total_failure(agent: CognitiveAgent, monkeypatch):
    monkeypatch.setattr(agent, "_get_interpretation", lambda *args: None)
    monkeypatch.setattr(agent, "_handle_cognitive_reflex", lambda *args: None)

    response = agent.chat("... gibberish ...")
    assert "I'm sorry, I was unable to understand that." in response


def test_get_interpretation_symbolic_success(agent: CognitiveAgent, monkeypatch):
    mock_interp = InterpretData(
        intent="unknown",
        entities=[],
        relation=None,
        key_topics=[],
        full_text_rephrased="",
    )
    monkeypatch.setattr(type(agent.parser), "parse", lambda self, text: [mock_interp])

    result = agent._get_interpretation("test input")
    assert result is not None
    assert result[0]["intent"] == "unknown"


def test_handle_cognitive_reflex_no_unknown_words(agent: CognitiveAgent, monkeypatch):
    monkeypatch.setattr(type(agent.lexicon), "is_known_word", lambda self, word: True)
    response = agent._handle_cognitive_reflex("all known words")
    assert response is None


def test_handle_cognitive_reflex_research_success_and_reentry(
    agent: CognitiveAgent, monkeypatch
):
    # First encounter of "unknown" returns False; subsequent checks return True.
    call_counts: dict[str, int] = {}

    def is_known_word_side_effect(self, word: str) -> bool:
        count = call_counts.get(word, 0)
        call_counts[word] = count + 1
        if word == "unknown":
            return count > 0
        return True

    monkeypatch.setattr(type(agent.lexicon), "is_known_word", is_known_word_side_effect)

    assert agent.harvester is not None
    monkeypatch.setattr(
        agent.harvester, "_resolve_investigation_goal", lambda goal: True, raising=False
    )
    monkeypatch.setattr(
        agent, "_chat_reentry_once", lambda text: "Re-entry successful."
    )

    response = agent._handle_cognitive_reflex("unknown word here")
    assert response == "Re-entry successful."


def test_learn_from_introspection_no_context(agent: CognitiveAgent, monkeypatch):
    """Test that introspection is skipped if no context can be derived."""
    mock_parse = MagicMock()
    monkeypatch.setattr(type(agent.parser), "parse", mock_parse)
    agent._learn_from_introspection(
        "response",
        InterpretData(
            intent="greeting",
            entities=[],
            relation=None,
            key_topics=[],
            full_text_rephrased="",
        ),
    )
    mock_parse.assert_not_called()


def test_resolve_references_no_pronouns(agent: CognitiveAgent):
    """Test that text without pronouns is returned unchanged."""
    text = "the sky is blue"
    assert agent._resolve_references(text) == text


def test_resolve_references_no_antecedent(agent: CognitiveAgent):
    """Test that pronouns are not resolved if no antecedent is in history."""
    agent.structured_history = []
    text = "what is it?"
    assert agent._resolve_references(text) == text


# --- Clarification & entity correction tests ---


def test_handle_clarification_full_flow(agent: CognitiveAgent, monkeypatch):
    """Test the full clarification logic from user input to knowledge update."""

    def get_edge(
        source: ConceptNode, target: ConceptNode, rel_type: str
    ) -> RelationshipEdge | None:
        for edge in agent.graph.get_edges_from_node(source.id):
            if edge.target == target.id and edge.type == rel_type:
                return edge
        return None

    agent.clarification_context = {"subject": "raven", "conflicting_relation": "is_a"}

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

    raven_node = agent.graph.add_node(ConceptNode("raven"))
    bird_node = agent.graph.add_node(ConceptNode("bird"))
    mammal_node = agent.graph.add_node(ConceptNode("mammal"))
    agent.graph.add_edge(raven_node, mammal_node, "is_a", weight=0.5)

    response = agent._handle_clarification("it is a bird")

    assert "Thank you for the clarification" in response
    bird_edge = get_edge(raven_node, bird_node, "is_a")
    mammal_edge = get_edge(raven_node, mammal_node, "is_a")

    assert bird_edge is not None
    assert bird_edge.weight == 1.0
    assert mammal_edge is not None
    assert mammal_edge.weight == 0.1


def test_get_corrected_entity_no_match(agent: CognitiveAgent):
    agent.graph.add_node(ConceptNode("testing"))
    assert agent._get_corrected_entity("tsting") == "testing"
    assert agent._get_corrected_entity("xyz") == "xyz"


def test_process_intent_unhandled_intent(agent: CognitiveAgent):
    interp = InterpretData(
        intent="unknown",
        entities=[],
        relation=None,
        key_topics=[],
        full_text_rephrased="",
    )
    response = agent._process_intent(interp, "some input")
    assert "not sure how to process that" in response


def test_find_specific_fact_multiple_results(agent: CognitiveAgent):
    agent_node = agent.graph.add_node(ConceptNode("agent"))
    ability1 = agent.graph.add_node(ConceptNode("learn"))
    ability2 = agent.graph.add_node(ConceptNode("reason"))
    agent.graph.add_edge(agent_node, ability1, "has_ability")
    agent.graph.add_edge(agent_node, ability2, "has_ability")

    response = agent._find_specific_fact("agent", "has_ability")
    assert response is not None
    assert "My abilities include:" in response
    assert "learn" in response
    assert "reason" in response


def test_answer_question_about_agent_name(agent: CognitiveAgent):
    agent_node = agent.graph.add_node(ConceptNode("agent"))
    name_node = agent.graph.add_node(ConceptNode("Axiom"))
    agent.graph.add_edge(agent_node, name_node, "has_name")
    response = agent._answer_question_about("agent", "what is your name?")
    assert "My name is Axiom" in response


def test_synthesize_response_non_trigger(agent: CognitiveAgent):
    response, was_used = agent._synthesize_response("Hello User.", "hello")
    assert response == "Hello User."
    assert was_used is False


def test_perform_multi_hop_query_max_hops(agent: CognitiveAgent):
    n1 = agent.graph.add_node(ConceptNode("n1"))
    n2 = agent.graph.add_node(ConceptNode("n2"))
    n3 = agent.graph.add_node(ConceptNode("n3"))
    n4 = agent.graph.add_node(ConceptNode("n4"))
    agent.graph.add_edge(n1, n2, "rel")
    agent.graph.add_edge(n2, n3, "rel")
    agent.graph.add_edge(n3, n4, "rel")
    path = agent._perform_multi_hop_query(n1, n4, max_hops=2)
    assert path is None


def test_format_path_as_sentence_empty_path(agent: CognitiveAgent):
    assert agent._format_path_as_sentence([]) == ""


def test_reboot_interpreter(agent: CognitiveAgent):
    agent.interpreter.synthesis_cache = {"test": "value"}
    old_interpreter_id = id(agent.interpreter)

    with patch("axiom.cognitive_agent.UniversalInterpreter"):
        agent._reboot_interpreter()
        new_interpreter_id = id(agent.interpreter)
        assert old_interpreter_id != new_interpreter_id
        # ensure old cache passed
        assert agent.interpreter.synthesis_cache.get("test") == "value"


def test_log_autonomous_cycle_completion_triggers_reboot(agent: CognitiveAgent):
    agent.autonomous_cycle_count = agent.INTERPRETER_REBOOT_THRESHOLD - 1
    with patch.object(agent, "_reboot_interpreter", lambda: None):
        agent.log_autonomous_cycle_completion()
        assert agent.autonomous_cycle_count == 0


def test_load_agent_state_corrupt_file(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{,}")
    with patch("axiom.cognitive_agent.UniversalInterpreter"):
        agent = CognitiveAgent(state_file=state_file)
        assert agent.learning_iterations == 0


# --- Facts & filtering ---


def test_gather_facts_multihop_filtering(agent: CognitiveAgent):
    start_node = agent.graph.add_node(ConceptNode("start"))
    for i in range(12):
        other_node = agent.graph.add_node(ConceptNode(f"node{i}"))
        edge = agent.graph.add_edge(start_node, other_node, "rel")
        if edge:
            agent.graph.graph.edges[edge.source, edge.target, edge.id][
                "access_count"
            ] = i

    facts = agent._gather_facts_multihop(start_node.id, max_hops=1)
    assert len(facts) == 10


def test_filter_facts_for_temporal_query(agent: CognitiveAgent):
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    facts_tuple = (
        ("fact1", (("effective_date", yesterday.isoformat()),)),
        ("fact2", (("effective_date", today.isoformat()),)),
        ("fact3", (("effective_date", tomorrow.isoformat()),)),
        ("fact4", (("some_prop", "value"),)),
    )
    facts_with_props: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = facts_tuple
    filtered = agent._filter_facts_for_temporal_query(facts_with_props)
    assert filtered == {"fact2"}


# --- Sentence sanitization & statement processing ---


def test_sanitize_sentence_for_learning(agent: CognitiveAgent):
    assert agent._sanitize_sentence_for_learning("Text (with parens).") == "Text ."
    assert agent._sanitize_sentence_for_learning("According to sources, text") == "text"
    assert agent._sanitize_sentence_for_learning(" - Text;") == "Text"


def test_process_statement_for_learning_incomplete_relation(agent: CognitiveAgent):
    relation = RelationData(subject="a", verb="is")
    learned, msg = agent._process_statement_for_learning(relation)
    assert not learned
    assert "Incomplete fact structure" in msg


# --- Conflict resolution & knowledge updates ---


def test_resolve_exclusive_conflict_stalemate(agent: CognitiveAgent):
    sub_node = agent.graph.add_node(ConceptNode("earth"))
    obj_node = agent.graph.add_node(ConceptNode("sphere"))
    existing_obj = agent.graph.add_node(ConceptNode("oblate spheroid"))
    agent.graph.add_edge(
        sub_node, existing_obj, "is_a", weight=0.95, properties={"provenance": "user"}
    )

    relation_data = RelationData(properties={"confidence": 0.95, "provenance": "user"})
    learned, msg = agent._resolve_exclusive_conflict(
        sub_node, obj_node, "is_a", relation_data
    )

    assert not learned
    assert msg in {"exclusive_conflict", "existing_fact_stronger"}
    # Depending on confidence/provenance logic, a follow-up goal may or may not be added.


def test_add_or_update_concept_quietly_empty_name(agent: CognitiveAgent):
    assert agent._add_or_update_concept_quietly("  ") is None


def test_manual_add_knowledge_quietly(agent: CognitiveAgent):
    agent.manual_add_knowledge_quietly("cat", "noun", "is_a", "animal")
    cat_node = agent.graph.get_node_by_name("cat")
    animal_node = agent.graph.get_node_by_name("animal")
    assert cat_node is not None
    assert animal_node is not None


def test_save_state(agent: CognitiveAgent, monkeypatch):
    monkeypatch.setattr(agent, "_save_agent_state", lambda: None)
    agent.save_state()


def test_learn_new_fact_autonomously_success(agent: CognitiveAgent, monkeypatch):
    """Test successful autonomous learning from a fact sentence."""
    mock_relations = [
        RelationData(
            subject="cat",
            verb="is_a",
            object="mammal",
            properties={"confidence": 0.8, "provenance": "test"},
        )
    ]

    monkeypatch.setattr(
        agent.interpreter,
        "decompose_sentence_to_relations",
        lambda text, main_topic: mock_relations,
    )
    monkeypatch.setattr(
        agent, "_process_statement_for_learning", lambda relation: (True, "success")
    )

    result = agent.learn_new_fact_autonomously("A cat is a mammal", "animals")
    assert result is True


def test_learn_new_fact_autonomously_no_relations(agent: CognitiveAgent, monkeypatch):
    """Test autonomous learning when interpreter returns no relations."""
    monkeypatch.setattr(
        agent.interpreter,
        "decompose_sentence_to_relations",
        lambda text, main_topic: [],
    )

    result = agent.learn_new_fact_autonomously("gibberish text", "topic")
    assert result is False


def test_learn_new_fact_autonomously_processing_fails(
    agent: CognitiveAgent, monkeypatch
):
    """Test autonomous learning when relation processing fails."""
    mock_relations = [
        RelationData(
            subject="invalid",
            verb="relation",
            object="data",
            properties={"confidence": 0.3, "provenance": "test"},
        )
    ]

    monkeypatch.setattr(
        agent.interpreter,
        "decompose_sentence_to_relations",
        lambda text, main_topic: mock_relations,
    )
    monkeypatch.setattr(
        agent, "_process_statement_for_learning", lambda relation: (False, "failed")
    )

    result = agent.learn_new_fact_autonomously("Invalid fact", "topic")
    assert result is False


def test_learn_new_fact_autonomously_sets_default_properties(
    agent: CognitiveAgent, monkeypatch
):
    """Test that autonomous learning sets default confidence and provenance."""
    captured_relations = []

    def capture_relation(relation: RelationData) -> tuple[bool, str]:
        captured_relations.append(relation)
        return True, "success"

    mock_relations = [RelationData(subject="test", verb="is_a", object="example")]

    monkeypatch.setattr(
        agent.interpreter,
        "decompose_sentence_to_relations",
        lambda text, main_topic: mock_relations,
    )
    monkeypatch.setattr(agent, "_process_statement_for_learning", capture_relation)

    result = agent.learn_new_fact_autonomously("Test is an example", "testing")
    assert result is True
    assert len(captured_relations) == 1
    assert captured_relations[0]["properties"]["confidence"] == 0.6
    assert captured_relations[0]["properties"]["provenance"] == "llm_decomposition"
