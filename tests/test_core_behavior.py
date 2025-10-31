from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_base import validate_and_add_relation

if TYPE_CHECKING:
    from axiom.universal_interpreter import PropertyData


@pytest.fixture
def agent(tmp_path: Path) -> CognitiveAgent:
    """Provides a fresh, non-learning CognitiveAgent for testing."""
    brain_file = tmp_path / "brain.json"
    state_file = tmp_path / "state.json"
    return CognitiveAgent(
        brain_file=brain_file,
        state_file=state_file,
        load_from_file=True,
        enable_llm=False,
    )


def test_validate_and_add_relation_defers_unknown_words(agent: CognitiveAgent):
    """
    Given a relation containing unknown words, validate_and_add_relation
    should return 'deferred' and create new INVESTIGATE goals.
    """
    assert not agent.lexicon.is_known_word("newwordify")

    relation_dict = {
        "subject": "newwordify",
        "verb": "is_a",
        "object": "flangdoodle",
    }
    properties = cast("PropertyData", {"confidence": 0.3, "provenance": "user"})

    status = validate_and_add_relation(agent, relation_dict, properties)

    assert status == "deferred"
    assert "INVESTIGATE: newwordify" in agent.learning_goals
    assert "INVESTIGATE: flangdoodle" in agent.learning_goals
