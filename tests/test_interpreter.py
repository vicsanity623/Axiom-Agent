from pathlib import Path
from unittest.mock import MagicMock

import pytest

from axiom.universal_interpreter import UniversalInterpreter


def test_interpreter_init_raises_error_for_missing_model(monkeypatch, tmp_path):
    """
    Covers the FileNotFoundError branch in UniversalInterpreter.__init__.
    Ensures the agent crashes gracefully if the LLM model file is missing.
    """
    non_existent_path = tmp_path / "this_model_does_not_exist.gguf"

    monkeypatch.setattr("axiom.universal_interpreter.Llama", MagicMock())

    with pytest.raises(FileNotFoundError, match="Interpreter model not found"):
        _ = UniversalInterpreter(model_path=non_existent_path, load_llm=True)

    print("Correctly raised FileNotFoundError for missing model file.")


def test_interpreter_graceful_failure_without_llm():
    """
    Tests that the UniversalInterpreter behaves correctly and does not crash
    when initialized with the LLM disabled.
    """
    interpreter = UniversalInterpreter(load_llm=False)
    assert interpreter.llm is None

    interpret_result = interpreter.interpret("what is a cat?")
    assert interpret_result["intent"] == "unknown"
    assert "(LLM disabled)" in interpret_result["full_text_rephrased"]

    original_input = "what about it?"
    context_result = interpreter.resolve_context(["User: a raven"], original_input)
    assert context_result == original_input

    breakdown_result = interpreter.break_down_definition("bird", "a feathered animal")
    assert breakdown_result == []

    questions_result = interpreter.generate_curious_questions(
        "bird",
        "a bird has wings",
    )
    assert questions_result == []

    synthesis_result = interpreter.synthesize("a bird is an animal")
    assert str(synthesis_result) == "a bird is an animal"


def test_interpreter_json_cleaning_and_caching(tmp_path: Path):
    """
    Tests the non-LLM utility functions of the interpreter.
    """
    cache_file = tmp_path / "test_cache.json"
    interpreter = UniversalInterpreter(load_llm=False, cache_file=cache_file)

    messy_json = (
        'Some garbage text before {"key": "value", "trailing": "comma",} and after.'
    )
    cleaned_json = interpreter._clean_llm_json_output(messy_json)
    assert cleaned_json == '{"key": "value", "trailing": "comma"}'
    assert interpreter._clean_llm_json_output("no json here") == ""

    interpreter.synthesis_cache["hello"] = "world"
    interpreter._save_cache()
    assert cache_file.exists()

    new_interpreter = UniversalInterpreter(load_llm=False, cache_file=cache_file)
    assert "hello" in new_interpreter.synthesis_cache
    assert new_interpreter.synthesis_cache["hello"] == "world"


def test_interpret_with_context_routing(monkeypatch):
    """
    Tests the routing logic of interpret_with_context.
    """
    resolve_was_called = False

    def mock_resolve_context(self, history, new_input):
        nonlocal resolve_was_called
        resolve_was_called = True
        return new_input

    monkeypatch.setattr(
        "axiom.universal_interpreter.UniversalInterpreter.resolve_context",
        mock_resolve_context,
    )

    interpreter = UniversalInterpreter(load_llm=False)
    history = ["User: What is a raven?"]

    resolve_was_called = False
    interpreter.interpret_with_context("what color is it?", history)
    assert resolve_was_called is True

    resolve_was_called = False
    interpreter.interpret_with_context("what is a bird?", history)
    assert resolve_was_called is False

    resolve_was_called = False
    interpreter.interpret_with_context("what color is it?", [])
    assert resolve_was_called is False


class MockFailingLlama:
    def __call__(self, *args, **kwargs):
        raise ValueError("Simulated LLM Crash")


def test_interpreter_handles_llm_call_failures():
    """
    Covers the `try...except` blocks in the UniversalInterpreter.
    Ensures that if the LLM call itself raises an exception, the interpreter
    returns a safe, default value instead of crashing the agent.
    """
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    interpret_result = interpreter.interpret("this will cause a crash")
    assert interpret_result["intent"] == "unknown"
    assert "Could not fully interpret" in interpret_result["full_text_rephrased"]
    print("interpret() correctly handled an LLM crash.")

    original_input = "what about it?"
    context_result = interpreter.resolve_context(["User: a raven"], original_input)
    assert context_result == original_input
    print("resolve_context() correctly handled an LLM crash.")

    breakdown_result = interpreter.break_down_definition("bird", "a feathered animal")
    assert breakdown_result == []
    print("break_down_definition() correctly handled an LLM crash.")

    questions_result = interpreter.generate_curious_questions(
        "bird",
        "a bird has wings",
    )
    assert questions_result == []
    print("generate_curious_questions() correctly handled an LLM crash.")

    synthesis_result = interpreter.synthesize("a bird is an animal")
    assert synthesis_result == "a bird is an animal"
    print("synthesize() correctly handled an LLM crash.")
