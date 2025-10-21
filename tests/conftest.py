# in tests/conftest.py
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

# Move all application imports into the TYPE_CHECKING block.
if TYPE_CHECKING:
    from pathlib import Path

    from axiom.cognitive_agent import CognitiveAgent
    from axiom.universal_interpreter import InterpretData


class MockUniversalInterpreter:
    """
    A fake UniversalInterpreter that behaves more realistically for tests.
    """

    def __init__(self, *args, **kwargs):
        self.llm = None
        print("--- Initialized MockUniversalInterpreter (No LLM Loaded) ---")

    def verify_and_reframe_fact(
        self,
        original_topic: str,
        raw_sentence: str,
    ) -> str | None:
        """A simple mock for the verification method."""
        if original_topic.lower() in raw_sentence.lower():
            return raw_sentence
        return None

    def synthesize(
        self,
        structured_facts: str | list,
        original_question: str | None = None,
        mode: str = "statement",
        **kwargs,
    ) -> str:
        """A mock synthesizer that correctly handles different modes and data types."""
        if mode == "clarification_question":
            return f"Which is correct regarding {structured_facts}?"
        if isinstance(structured_facts, list):
            fact_strings = []
            for fact_tuple in structured_facts:
                if len(fact_tuple) == 3:
                    subject, relation, obj = fact_tuple
                    fact_strings.append(f"{subject} {relation.replace('_', ' ')} {obj}")
            if fact_strings:
                return ". ".join(fact_strings) + "."
        return str(structured_facts)

    def interpret(self, user_input: str) -> InterpretData | None:
        # Must import here because the type is only available in TYPE_CHECKING
        from axiom.universal_interpreter import InterpretData

        if "gibberish" in user_input:
            return None
        return InterpretData(
            intent="unknown",
            entities=[],
            relation=None,
            key_topics=[],
            full_text_rephrased="",
        )


@pytest.fixture
def agent(monkeypatch, tmp_path: Path) -> CognitiveAgent:
    """
    A globally available fixture that creates a fresh, clean CognitiveAgent.
    """
    # We must import the real class at runtime inside the fixture
    from axiom.cognitive_agent import CognitiveAgent

    monkeypatch.setattr(
        "axiom.cognitive_agent.UniversalInterpreter",
        MockUniversalInterpreter,
    )
    brain_file = tmp_path / "test_brain.json"
    state_file = tmp_path / "test_state.json"
    return CognitiveAgent(brain_file=brain_file, state_file=state_file)


@pytest.fixture
def blank_agent(monkeypatch, tmp_path: Path) -> CognitiveAgent:
    """
    A fixture that provides a CognitiveAgent with a COMPLETELY EMPTY brain.
    """
    # We must import the real class at runtime inside the fixture
    from axiom.cognitive_agent import CognitiveAgent

    monkeypatch.setattr(
        "axiom.cognitive_agent.UniversalInterpreter",
        MockUniversalInterpreter,
    )
    monkeypatch.setattr(
        "axiom.cognitive_agent.seed_domain_knowledge",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "axiom.cognitive_agent.seed_core_vocabulary",
        lambda *args: None,
    )

    brain_file = tmp_path / "blank_brain.json"
    state_file = tmp_path / "blank_state.json"
    return CognitiveAgent(brain_file=brain_file, state_file=state_file)
