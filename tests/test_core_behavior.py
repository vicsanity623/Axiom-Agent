from axiom.cognitive_agent import CognitiveAgent


def make_agent(tmp_path):
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "state.json"
    # If file doesn't exist, CognitiveAgent will create a fresh graph (and seed).
    return CognitiveAgent(
        brain_file=brain_file,
        state_file=state_file,
        load_from_file=True,
        enable_llm=False,
    )


def test_lexicon_promotion_and_defer(tmp_path):
    agent = make_agent(tmp_path)
    # Use lexicon observe to simulate parser seeing a word used as a verb multiple times
    word = "flim"
    for _ in range(4):
        agent.lexicon.observe_word_pos(word, "verb", confidence=0.5)
    # Attempt to promote (same logic used by code automatically)
    promoted = (
        agent.graph.graph.nodes[agent.graph.get_node_by_name(word).id]
        .get("properties", {})
        .get("lexical_promoted_as", None)
    )
    # Promotion should happen eventually (threshold 0.8 required)
    assert promoted == "verb"

    # Now assert a fact using that new verb will be accepted (no longer deferred)
    agent.chat("Bloop flim Blap.")
    assert agent.graph.get_node_by_name("bloop") is not None
    assert agent.graph.get_node_by_name("blap") is not None


def test_deferred_insertion_low_confidence(tmp_path):
    """
    Statements with unknown, low-confidence words should still be deferred.
    """
    agent = make_agent(tmp_path)

    # GIVEN: Unknown words (no lexical data)
    assert agent.graph.get_node_by_name("newwordify") is None
    assert agent.graph.get_node_by_name("flangdoodle") is None

    # WHEN: The agent sees a low-confidence fact
    relation = {
        "subject": {"name": "newwordify"},
        "verb": "is_a",
        "object": {"name": "flangdoodle"},
        "properties": {"confidence": 0.3, "provenance": "user"},
    }
    ok, msg = agent._process_statement_for_learning(relation)

    # THEN: The relation should be deferred
    assert ok is False
    assert msg in ("deferred", "exclusive_conflict", "exclusive_conflict")

    src = agent.graph.get_node_by_name("newwordify")
    tgt = agent.graph.get_node_by_name("flangdoodle")

    assert src is not None
    assert tgt is not None

    edges = agent.graph.get_edges_from_node(src.id)
    assert all(e.target != tgt.id for e in edges), "Relation should be deferred."


def test_auto_promotion_insertion_high_confidence(tmp_path):
    """
    High-confidence definitional statements should auto-promote and be inserted.
    """
    agent = make_agent(tmp_path)

    # GIVEN: Unknown words initially
    assert agent.graph.get_node_by_name("blorptufts") is None
    assert agent.graph.get_node_by_name("feathermass") is None

    # WHEN: The agent learns a high-confidence definitional fact
    relation = {
        "subject": {"name": "blorptufts"},
        "verb": "is_a",
        "object": {"name": "feathermass"},
        "properties": {"confidence": 0.95, "provenance": "llm_verified"},
    }
    ok, msg = agent._process_statement_for_learning(relation)

    # THEN: The relation should be inserted, not deferred
    assert ok is True, f"Expected insertion, got {msg}"

    src = agent.graph.get_node_by_name("blorptufts")
    tgt = agent.graph.get_node_by_name("feathermass")
    assert src is not None
    assert tgt is not None

    edges = agent.graph.get_edges_from_node(src.id)
    found = any(e.target == tgt.id and e.type == "is_a" for e in edges)
    assert found, "Expected blorptufts --[is_a]--> feathermass to exist."


def test_contradiction_storage(tmp_path):
    agent = make_agent(tmp_path)
    # Add a base fact as user (high confidence)
    agent.chat("Paris is the capital of France.")
    # Now add contradictory fact with low confidence
    relation = {
        "subject": "paris",
        "verb": "is_capital_of",
        "object": "germany",
        "properties": {"confidence": 0.6, "provenance": "llm"},
    }

    from axiom.knowledge_base import validate_and_add_relation

    status = validate_and_add_relation(
        agent,
        relation,
        {"confidence": 0.6, "source": "llm"},
    )

    assert status in ("contradiction_stored", "deferred", "inserted")
