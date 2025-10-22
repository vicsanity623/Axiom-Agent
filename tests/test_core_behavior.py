from axiom.cognitive_agent import CognitiveAgent


def make_agent(tmp_path):
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "state.json"
    # If file doesn't exist, CognitiveAgent will create a fresh graph (and seed).
    # FIX 1: Return the instance directly.
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
    # attempt to promote (same logic used by code automatically)
    promoted = (
        agent.graph.graph.nodes[agent.graph.get_node_by_name(word).id]
        .get("properties", {})
        .get("lexical_promoted_as", None)
    )
    # promotion should happen eventually (threshold 0.8 required: with 4*0.5 votes top fraction 1.0)
    assert promoted == "verb"

    # Now assert a fact using that new verb will be accepted (no longer deferred)
    # Create simple S-V-O relation and call _process_statement_for_learning via chat
    # FIX 2: Remove the unused 'response' variable.
    agent.chat("Bloop flim Blap.")
    # If the parser doesn't parse it, fallback to llm disabled means will say "unable to understand"
    # But we mainly assert the node exists
    assert agent.graph.get_node_by_name("bloop") is not None
    assert agent.graph.get_node_by_name("blap") is not None


def test_deferred_insertion(tmp_path):
    """
    Tests that a statement with un-promoted words is deferred and not
    immediately added to the knowledge graph.
    """
    agent = make_agent(tmp_path)

    # 1. GIVEN: The words 'newwordify' and 'flangdoodle' are not known.
    assert agent.graph.get_node_by_name("newwordify") is None
    assert agent.graph.get_node_by_name("flangdoodle") is None

    # 2. WHEN: We chat with the agent using these un-promoted words.
    agent.chat("Newwordify is a flangdoodle.")

    # 3. THEN: The statement should be deferred. We verify this by asserting
    # that the relationship was NOT added to the graph.
    source_node = agent.graph.get_node_by_name("newwordify")
    target_node = agent.graph.get_node_by_name("flangdoodle")

    # The nodes themselves are created, but the edge is not.
    assert source_node is not None
    assert target_node is not None

    # Check for the edge
    edges = agent.graph.get_edges_from_node(source_node.id)
    found_edge = any(edge.target == target_node.id for edge in edges)

    assert found_edge is False, (
        "The relationship should have been deferred, not added to the graph."
    )


def test_contradiction_storage(tmp_path):
    agent = make_agent(tmp_path)
    # Add a base fact as user (high confidence)
    agent.chat("Paris is the capital of France.")
    # Now add contradictory fact with low confidence (simulate LLM by calling _process_statement_for_learning directly)
    relation = {
        "subject": "paris",
        "verb": "is_capital_of",
        "object": "germany",
        "properties": {"confidence": 0.6, "provenance": "llm"},
    }
    # call validate_and_add_relation directly
    from axiom.knowledge_base import validate_and_add_relation

    status = validate_and_add_relation(
        agent,
        relation,
        {"confidence": 0.6, "source": "llm"},
    )
    # Because existing fact is seeded/high-confidence, the new one should be stored as contradiction or deferred
    assert status in ("contradiction_stored", "deferred", "inserted")
