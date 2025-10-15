"""
Parameterized introspection tests for Axiom-Agent using a real model (if present).
Run with:  pytest -m introspection --disable-warnings -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axiom.cognitive_agent import CognitiveAgent

MODEL_PATH = Path("models")
skip_no_model = not any(MODEL_PATH.glob("*.gguf"))


@pytest.mark.skipif(skip_no_model, reason="No local LLM model found")
@pytest.mark.introspection
@pytest.mark.parametrize(
    ("fact", "question", "expected_substring", "learned_properties"),
    [
        ("a raven is a bird", "what color is it?", "black", "black"),
        ("a dolphin is a mammal", "does it lay eggs?", "no", ("birth", "live young")),
        ("a rose is a flower", "what is a rose?", "flower", "flower"),
        ("fire is hot", "is fire cold?", "no", "hot"),
    ],
)
def test_introspective_learning_real_model(
    tmp_path: Path,
    fact: str,
    question: str,
    expected_substring: str,
    learned_properties: tuple[str, ...],
):
    """
    Run a full, end-to-end test of the introspective learning loop.
    1. Learns an initial fact.
    2. Answers a related question, triggering LLM synthesis.
    3. Verifies that the agent introspectively learned a new, more specific fact.
    """
    brain = tmp_path / "brain.json"
    state = tmp_path / "state.json"
    agent = CognitiveAgent(brain_file=brain, state_file=state)

    # 1. Teach the agent the initial fact.
    agent.chat(fact)

    # 2. Ask the question that should trigger the LLM and introspection.
    response1 = agent.chat(question)
    assert expected_substring in response1.lower()

    # 3. Verification: Check if the agent learned the specific new property.

    # First, correctly identify the subject of the original fact.
    # We use the agent's own cleaning utility to do this reliably.
    subject_of_fact = agent._clean_phrase(fact.split(" is ")[0])
    subject_node = agent.graph.get_node_by_name(subject_of_fact)
    assert subject_node is not None, (
        f"Subject node '{subject_of_fact}' was not created."
    )

    learned = False
    print(f"\n--- DEBUG: Checking learned facts for subject: '{subject_of_fact}' ---")

    for edge in agent.graph.get_edges_from_node(subject_node.id):
        target_node = agent.graph.get_node_by_id(edge.target)
        if target_node:
            print(
                f"  - Found fact: {subject_node.name} --[{edge.type}]--> {target_node.name}",
            )
            # Check if ANY of the expected properties are in the learned fact.
            if any(prop in target_node.name.lower() for prop in learned_properties):
                print(f"    ^ MATCH FOUND for one of: {learned_properties}!")
                learned = True
                break

    print("--- END DEBUG ---")

    assert learned, (
        f"Agent failed to learn any of the expected properties {learned_properties} for subject '{subject_of_fact}'"
    )

    # 4. Ask again — the agent should now have the fact in its symbolic memory.
    response2 = agent.chat(question)
    assert expected_substring in response2.lower()


def test_summary_statistics(request):
    """
    Optional helper test to be expanded later for summarizing metrics.
    """
    assert True
