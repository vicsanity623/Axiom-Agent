# in tests/conftest.py
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from axiom.universal_interpreter import InterpretData

if TYPE_CHECKING:
    from pathlib import Path

    from axiom.cognitive_agent import CognitiveAgent


class MockUniversalInterpreter:
    """
    A fake UniversalInterpreter that behaves more realistically for tests.
    """

    def __init__(self, *args, **kwargs):
        self.llm = None
        print("--- Initialized MockUniversalInterpreter (No LLM Loaded) ---")
        pass

    def synthesize(
        self,
        structured_facts: str | list,
        original_question: str | None = None,
        mode: str = "statement",
        **kwargs,
    ) -> str:
        """
        A mock synthesizer that correctly handles different modes and data types.
        """
        if mode == "clarification_question":
            return f"Which is correct regarding {structured_facts}?"

        # If the input is a list of facts (as it is in _answer_question_about),
        # format them into a simple sentence for the test to check.
        if isinstance(structured_facts, list):
            # This mimics the real synthesizer's job of turning structured data into a sentence.
            # Example input: [('canary', 'is_a', 'bird'), ('canary', 'has_property', 'yellow')]
            fact_strings = []
            for fact_tuple in structured_facts:
                if len(fact_tuple) == 3:  # Simple (subject, relation, object) tuple
                    subject, relation, obj = fact_tuple
                    fact_strings.append(f"{subject} {relation.replace('_', ' ')} {obj}")

            if fact_strings:
                return ". ".join(fact_strings) + "."

        # Fallback for simple string inputs
        return str(structured_facts)

    def interpret(self, user_input: str) -> InterpretData | None:
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
    A globally available fixture that creates a fresh, clean CognitiveAgent
    for every test. It automatically uses the mock interpreter.
    """

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
    It does this by patching the brain-seeding functions to do nothing.
    """

    from axiom.cognitive_agent import CognitiveAgent

    # Temporarily replace the real, slow interpreter with our fake one.
    monkeypatch.setattr(
        "axiom.cognitive_agent.UniversalInterpreter",
        MockUniversalInterpreter,
    )
    # Temporarily replace the brain-seeding functions with dummies.
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

    # When this agent is created, the seeding functions will be the dummies,
    # so the brain will remain empty.
    return CognitiveAgent(brain_file=brain_file, state_file=state_file)
