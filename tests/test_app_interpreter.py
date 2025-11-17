import json
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from axiom.universal_interpreter import (
    InterpretData,
    UniversalInterpreter,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    """A fixture to create a mock Llama instance."""
    return MagicMock()


@pytest.fixture
def interpreter_no_llm() -> UniversalInterpreter:
    """An interpreter instance with the LLM explicitly not loaded."""
    return UniversalInterpreter(load_llm=False)


@pytest.fixture
def interpreter_with_mock_llm(mock_llm: MagicMock) -> UniversalInterpreter:
    """An interpreter instance with a mocked LLM, bypassing the actual model loading."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = mock_llm
    return interpreter


def test_init_no_llm() -> None:
    """
    Given: The UniversalInterpreter is initialized.
    When: `load_llm` is set to False.
    Then: The `llm` attribute should be None.
    """
    interpreter = UniversalInterpreter(load_llm=False)
    assert interpreter.llm is None


def test_init_llm_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Given: The path to an LLM model does not exist.
    When: The UniversalInterpreter is initialized with `load_llm=True`.
    Then: A FileNotFoundError should be raised.
    """
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError):
        UniversalInterpreter(model_path="non_existent_model.gguf", load_llm=True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("what color is it?", True),
        ("tell me about them", True),
        ("this is a test", False),
        ("their idea was great", True),
        ("the idea was great", False),
        ("bither", False),
    ],
)
def test_is_pronoun_present(
    interpreter_no_llm: UniversalInterpreter, text: str, expected: bool
) -> None:
    """
    Given: An interpreter instance.
    When: `_is_pronoun_present` is called with various texts.
    Then: It should correctly identify the presence of whole-word pronouns.
    """
    assert interpreter_no_llm._is_pronoun_present(text) is expected


def test_cache_loading(tmp_path: Path) -> None:
    """
    Given: A cache file exists with valid data.
    When: The interpreter is initialized.
    Then: The caches should be loaded from the file.
    """
    cache_file = tmp_path / "cache.json"
    cache_data = {
        "interpretations": [["hello", {"intent": "greeting"}]],
        "synthesis": [["key", "value"]],
    }
    cache_file.write_text(json.dumps(cache_data))

    interpreter = UniversalInterpreter(load_llm=False, cache_file=cache_file)
    assert interpreter.interpretation_cache["hello"] == {"intent": "greeting"}
    assert interpreter.synthesis_cache["key"] == "value"


def test_cache_saving(tmp_path: Path) -> None:
    """
    Given: An interpreter with data in its caches.
    When: `_save_cache` is called.
    Then: The cache data should be written to the specified file.
    """
    cache_file = tmp_path / "cache.json"
    interpreter = UniversalInterpreter(load_llm=False, cache_file=cache_file)
    interpreter.interpretation_cache["test"] = cast(
        "InterpretData", {"intent": "unknown"}
    )
    interpreter._save_cache()

    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert data["interpretations"][0][0] == "test"


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ('Some text {"key": "value"} more text', '{"key": "value"}'),
        ('{"key": "value",}', '{"key": "value"}'),
        ('[{"key": "value"},]', '[{"key": "value"}]'),
        ("No JSON here", ""),
    ],
)
def test_clean_llm_json_output(
    interpreter_no_llm: UniversalInterpreter, raw_text: str, expected: str
) -> None:
    """
    Given: Various raw text outputs from an LLM.
    When: `_clean_llm_json_output` is called.
    Then: It should extract and clean the JSON string correctly.
    """
    cleaned = interpreter_no_llm._clean_llm_json_output(raw_text)
    assert cleaned == expected


def test_run_llm_completion_with_retries_success(
    interpreter_with_mock_llm: UniversalInterpreter,
) -> None:
    """
    Given: A mocked LLM that returns a valid response.
    When: `_run_llm_completion_with_retries` is called.
    Then: It should return the text from the LLM's first choice.
    """
    mock_response = {"choices": [{"text": "  response text  "}]}
    # FIX: Cast the llm attribute to MagicMock to satisfy mypy
    mock_llm = cast("MagicMock", interpreter_with_mock_llm.llm)
    mock_llm.return_value = mock_response

    result = interpreter_with_mock_llm._run_llm_completion_with_retries("prompt", 100)
    assert result == "response text"
    mock_llm.assert_called_once()


def test_run_llm_completion_with_retries_failure(
    interpreter_with_mock_llm: UniversalInterpreter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: A mocked LLM that consistently raises an error.
    When: `_run_llm_completion_with_retries` is called.
    Then: It should attempt to retry and ultimately return an empty string.
    """
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # FIX: Cast the llm attribute to MagicMock to satisfy mypy
    mock_llm = cast("MagicMock", interpreter_with_mock_llm.llm)
    mock_llm.side_effect = ValueError("LLM Error")

    result = interpreter_with_mock_llm._run_llm_completion_with_retries("prompt", 100)
    assert result == ""
    assert mock_llm.call_count == 3


def test_interpret_llm_disabled(interpreter_no_llm: UniversalInterpreter) -> None:
    """
    Given: An interpreter with the LLM disabled.
    When: `interpret` is called.
    Then: It should return a default 'unknown' interpretation.
    """
    result = interpreter_no_llm.interpret("test input")
    assert result["intent"] == "unknown"
    assert "LLM disabled" in result["full_text_rephrased"]


def test_interpret_success_with_negation(
    interpreter_with_mock_llm: UniversalInterpreter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: A mocked LLM that returns a fact containing negation.
    When: `interpret` is called.
    Then: The 'negated' property should be set to True and the text cleaned.
    """
    llm_output = '{"intent": "statement_of_fact", "relation": {"subject": "sky", "verb": "is not", "object": "green"}}'
    # FIX: Use monkeypatch.setattr to safely mock the method
    monkeypatch.setattr(
        interpreter_with_mock_llm,
        "_run_llm_completion_with_retries",
        lambda *args, **kwargs: llm_output,
    )

    result = interpreter_with_mock_llm.interpret("The sky is not green")
    relation = result.get("relation")
    assert relation is not None
    properties = relation.get("properties")
    assert properties is not None
    assert properties.get("negated") is True
    assert relation.get("verb") == "is"
    assert relation.get("object") == "green"


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ("simple string", "simple string"),
        ({"name": "test name"}, "test name"),
        ({"value": "test value"}, "test value"),
        ({"a": "foo", "b": {"name": "bar"}}, "foo bar"),
        (["one", {"name": "two"}], "one two"),
        ({"type": "ignored", "value": "kept"}, "kept"),
    ],
)
def test_flatten_structured_object(
    interpreter_no_llm: UniversalInterpreter, obj: Any, expected: str
) -> None:
    """
    Given: Various nested or simple data structures.
    When: `_flatten_structured_object` is called.
    Then: It should correctly flatten them into a single string.
    """
    assert interpreter_no_llm._flatten_structured_object(obj) == expected


def test_interpret_with_context_calls_resolver(
    interpreter_with_mock_llm: UniversalInterpreter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: An input with a pronoun and a conversation history.
    When: `interpret_with_context` is called.
    Then: It should call `resolve_context` before calling `interpret`.
    """
    mock_resolve = MagicMock(return_value="what color is the sky?")
    mock_interpret = MagicMock()
    monkeypatch.setattr(interpreter_with_mock_llm, "resolve_context", mock_resolve)
    monkeypatch.setattr(interpreter_with_mock_llm, "interpret", mock_interpret)

    interpreter_with_mock_llm.interpret_with_context(
        "what color is it?", ["User: what is the sky?"]
    )

    mock_resolve.assert_called_once_with(
        ["User: what is the sky?"], "what color is it?"
    )
    mock_interpret.assert_called_once_with("what color is the sky?")


def test_decompose_sentence_handles_malformed_llm_output(
    interpreter_with_mock_llm: UniversalInterpreter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: A mocked LLM that returns non-JSON text.
    When: `decompose_sentence_to_relations` is called.
    Then: It should handle the error gracefully and return an empty list.
    """
    # FIX: Use monkeypatch.setattr to safely mock the method
    monkeypatch.setattr(
        interpreter_with_mock_llm,
        "_run_llm_completion_with_retries",
        lambda *args, **kwargs: "Sorry, I can't do that.",
    )
    result = interpreter_with_mock_llm.decompose_sentence_to_relations("test")
    assert result == []


def test_synthesize_uses_cache(interpreter_with_mock_llm: UniversalInterpreter) -> None:
    """
    Given: A cache entry exists for a synthesis request.
    When: `synthesize` is called with the same arguments.
    Then: The cached value should be returned without calling the LLM.
    """
    facts = ["fact1"]
    question = "question"
    mode = "statement"
    cache_key = f"{mode}|{question}|{json.dumps([str(f) for f in facts])}"
    interpreter_with_mock_llm.synthesis_cache[cache_key] = "cached response"

    result = interpreter_with_mock_llm.synthesize(
        facts, original_question=question, mode=mode
    )

    assert result == "cached response"
    # FIX: Cast the llm attribute to MagicMock to satisfy mypy
    cast("MagicMock", interpreter_with_mock_llm.llm).assert_not_called()


def test_synthesize_clarification_mode(
    interpreter_with_mock_llm: UniversalInterpreter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The synthesizer is called in 'clarification_question' mode.
    When: The LLM is invoked.
    Then: The prompt should contain the specific instructions for that mode.
    """
    # --- FIX: Clear the cache to ensure the test is isolated ---
    interpreter_with_mock_llm.synthesis_cache.clear()

    mock_run = MagicMock(return_value="A question?")
    monkeypatch.setattr(
        interpreter_with_mock_llm, "_run_llm_completion_with_retries", mock_run
    )

    interpreter_with_mock_llm.synthesize("conflicts", mode="clarification_question")

    # The mock should now have been called
    assert mock_run.call_args is not None, "The mocked LLM function was never called"

    prompt_arg = mock_run.call_args[0][0]
    assert "You are an inquisitive AI agent" in prompt_arg
    assert "Conflicting Facts" in prompt_arg


def test_generate_curriculum_llm_disabled(
    interpreter_no_llm: UniversalInterpreter,
) -> None:
    """
    Given: An interpreter with the LLM disabled.
    When: `generate_curriculum` is called.
    Then: It should return an empty list.
    """
    assert interpreter_no_llm.generate_curriculum("test") == []


def test_generate_new_creative_seed_llm_disabled(
    interpreter_no_llm: UniversalInterpreter,
) -> None:
    """
    Given: An interpreter with the LLM disabled.
    When: `generate_new_creative_seed` is called.
    Then: It should return a hardcoded fallback seed.
    """
    assert "kitchen" in interpreter_no_llm.generate_new_creative_seed()
