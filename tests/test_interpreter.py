# in tests/test_interpreter.py

from pathlib import Path

from axiom.universal_interpreter import UniversalInterpreter


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
    # We use a "spy" to see if resolve_context was called.
    resolve_was_called = False

    def mock_resolve_context(self, history, new_input):
        nonlocal resolve_was_called
        resolve_was_called = True
        return new_input  # Return input unchanged for simplicity

    # Patch the CLASS, which is the correct way to use monkeypatch here.
    monkeypatch.setattr(
        "axiom.universal_interpreter.UniversalInterpreter.resolve_context",
        mock_resolve_context,
    )

    # Create an instance AFTER patching
    interpreter = UniversalInterpreter(load_llm=False)
    history = ["User: What is a raven?"]

    # Case A: Input has a pronoun, should call our spy.
    resolve_was_called = False  # Reset spy
    interpreter.interpret_with_context("what color is it?", history)
    assert resolve_was_called is True

    # Case B: Input has NO pronoun, should NOT call our spy.
    resolve_was_called = False  # Reset spy
    interpreter.interpret_with_context("what is a bird?", history)
    assert resolve_was_called is False

    # Case C: History is empty, should NOT call our spy.
    resolve_was_called = False  # Reset spy
    interpreter.interpret_with_context("what color is it?", [])
    assert resolve_was_called is False
