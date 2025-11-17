from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import spacy

from axiom.cognitive_agent import CognitiveAgent

if TYPE_CHECKING:
    from collections.abc import Generator

    from spacy.language import Language

    from axiom.universal_interpreter import InterpretData


class MockUniversalInterpreter:
    """A fake UniversalInterpreter that routes calls to realistic canned responses."""

    def __init__(self, *args, **kwargs):
        self.llm = "This is a mock LLM object"  # Make it not None
        print("--- Initialized MockUniversalInterpreter (No LLM Loaded) ---")

    def generate_curriculum(self, high_level_goal: str, **kwargs) -> list[str]:
        if "birds" in high_level_goal.lower():
            return ["bird", "feather", "nest"]
        return ["test topic 1", "test topic 2"]

    def verify_and_reframe_fact(
        self, original_topic: str, raw_sentence: str
    ) -> str | None:
        if original_topic.lower() in raw_sentence.lower():
            return raw_sentence.replace(".", "")  # Return a clean sentence
        return None

    def synthesize(
        self,
        structured_facts: str | list,
        original_question: str | None = None,
        **kwargs,
    ) -> str:
        # --- FIX: Add routing logic for synthesis tests ---
        if original_question and "What is a cat?" in original_question:
            return "Based on the facts, a cat is an animal."
        if isinstance(structured_facts, list) and "fact1" in structured_facts:
            return "Fact1 and Fact2 are related."
        if (
            isinstance(structured_facts, str)
            and "a cat is an animal" in structured_facts
        ):
            return "A cat is an animal"

        # Fallback for other tests
        if isinstance(structured_facts, list):
            return ". ".join(map(str, structured_facts)) + "."
        return str(structured_facts)

    def interpret(self, user_input: str) -> InterpretData | None:
        from axiom.universal_interpreter import InterpretData

        if "gibberish" in user_input:
            return None
        # A more realistic response for other tests
        return InterpretData(
            intent="statement_of_fact",
            entities=[],
            relation={"subject": "a", "verb": "is", "object": "b"},
            key_topics=["a", "b"],
            full_text_rephrased="A is B.",
        )

    def decompose_sentence_to_relations(self, *args, **kwargs) -> list:
        return [{"subject": "mock", "verb": "is", "object": "mocked"}]

    def resolve_context(self, history: list[str], new_input: str) -> str:
        return new_input


@pytest.fixture(scope="session")
def session_agent(monkeypatch_session) -> Generator[CognitiveAgent, None, None]:
    """
    A session-scoped fixture that creates a single, shared CognitiveAgent.
    This is the primary mechanism for speeding up the test suite.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        brain_file = Path(tmpdir) / "session_brain.json"
        state_file = Path(tmpdir) / "session_state.json"

        monkeypatch_session.setattr(
            "axiom.cognitive_agent.UniversalInterpreter",
            MockUniversalInterpreter,
        )

        agent = CognitiveAgent(
            brain_file=brain_file, state_file=state_file, enable_llm=False
        )
        yield agent


@pytest.fixture
def agent(session_agent: CognitiveAgent) -> CognitiveAgent:
    """
    A function-scoped fixture that provides a CLEAN agent for each test.
    It uses the fast session_agent as a base and simply resets its state,
    avoiding the slow re-initialization process.
    """
    session_agent.learning_goals.clear()
    session_agent.high_priority_tasks.clear()
    session_agent.goal_manager.goals.clear()
    if session_agent.harvester:
        session_agent.harvester.researched_terms.clear()

    return session_agent


@pytest.fixture
def blank_agent(session_agent: CognitiveAgent, monkeypatch) -> CognitiveAgent:
    """
    Provides a CognitiveAgent with a COMPLETELY EMPTY brain for specific tests.
    This is now much faster as it reuses the session agent's components.
    """
    monkeypatch.setattr(session_agent, "graph", MagicMock())

    session_agent.learning_goals.clear()
    session_agent.high_priority_tasks.clear()
    session_agent.goal_manager.goals.clear()

    return session_agent


@pytest.fixture(scope="session")
def spacy_nlp_lg() -> Language | None:
    """A session-scoped fixture that loads the large spaCy model once."""
    try:
        return spacy.load("en_core_web_lg")
    except (OSError, ImportError):
        print(
            "Warning: spaCy model 'en_core_web_lg' not found. spaCy-dependent tests will be skipped."
        )
        return None


@pytest.fixture(scope="session")
def monkeypatch_session() -> Generator[pytest.MonkeyPatch, None, None]:
    """A session-scoped monkeypatch fixture."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
