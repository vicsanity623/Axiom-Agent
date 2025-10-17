# in tests/test_parser.py

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from axiom.cognitive_agent import CognitiveAgent


@pytest.mark.introspection
@pytest.mark.parametrize(
    ("sentence", "expected_subject", "expected_relation", "expected_object"),
    [
        ("Paris is a city in France", "paris", "is_located_in", "france"),
        ("a wheel is part of a car", "wheel", "is_part_of", "car"),
        ("the book is on the table", "book", "is_on_top_of", "table"),
    ],
)
def test_preposition_parsing(
    tmp_path: Path,
    sentence,
    expected_subject,
    expected_relation,
    expected_object,
):
    """Test that the SymbolicParser correctly parses prepositional phrases."""
    # Setup a clean agent
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "state.json"
    agent = CognitiveAgent(
        brain_file=brain_file,
        state_file=state_file,
        enable_llm=False,
    )

    # Parse the sentence
    interpretations = agent.parser.parse(sentence)

    assert interpretations is not None, "Parser failed to produce any interpretation."
    assert len(interpretations) == 1, (
        "Parser produced more than one interpretation for a simple sentence."
    )

    interpretation = interpretations[0]
    relation = interpretation.get("relation")

    assert relation is not None, "Parser failed to extract a relation."

    # Verify the parsed components
    assert relation.get("subject") == expected_subject
    assert relation.get("verb") == expected_relation
    assert relation.get("object") == expected_object
