import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_interpreter_graceful_failure_without_llm(tmp_path: Path):
    """
    Tests that the UniversalInterpreter behaves correctly and does not crash
    when initialized with the LLM disabled, ensuring test isolation.
    """
    isolated_cache_file = tmp_path / "isolated_cache.json"
    interpreter = UniversalInterpreter(load_llm=False, cache_file=isolated_cache_file)
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


class MockWorkingLlama:
    def __call__(self, prompt, max_tokens, stop, echo, temperature):
        # FIX: More robust routing based on unique phrases in prompts.
        if "factual analysis engine" in prompt:  # interpret()
            return {
                "choices": [
                    {
                        "text": '{"intent": "statement_of_fact", "entities": [{"name": "cat", "type": "CONCEPT"}], "relation": {"subject": "cat", "verb": "is", "object": "animal"}, "key_topics": ["cat"], "full_text_rephrased": "A cat is an animal."}'
                    }
                ]
            }
        if (
            "knowledge engineering system" in prompt
        ):  # decompose_sentence_to_relations()
            return {
                "choices": [
                    {"text": '[{"subject": "cat", "verb": "is_a", "object": "animal"}]'}
                ]
            }
        if "logical decomposition engine" in prompt:  # break_down_definition()
            return {"choices": [{"text": "- Cat is a mammal\n- Cat is a pet"}]}
        if (
            "inquisitive assistant that thinks like a curious child" in prompt
        ):  # generate_curious_questions()
            return {
                "choices": [{"text": "- What do cats eat?\n- How long do cats live?"}]
            }
        if (
            "fact verification and reframing engine" in prompt
        ):  # verify_and_reframe_fact()
            return {"choices": [{"text": "Cat is a domestic animal"}]}
        if "language rephrasing engine" in prompt:  # synthesize() statement mode
            return {"choices": [{"text": "A cat is an animal"}]}
        if "curriculum design expert" in prompt:  # generate_curriculum()
            return {"choices": [{"text": "mammal, pet, domestic, feline, carnivore"}]}
        if "coreference resolution engine" in prompt:  # resolve_context()
            return {"choices": [{"text": "what color is an apple?"}]}
        if "inquisitive AI agent" in prompt:  # synthesize() clarification mode
            return {"choices": [{"text": "Which of these is correct about cats?"}]}
        # Default fallback for any unmatched prompt
        return {"choices": [{"text": "Default mock response."}]}


def test_interpreter_init_with_llm(tmp_path):
    """Test initialization with LLM enabled."""
    model_path = tmp_path / "test_model.gguf"
    model_path.touch()  # Create empty file

    with patch("axiom.universal_interpreter.Llama") as mock_llama_class:
        mock_llama_instance = MagicMock()
        mock_llama_class.return_value = mock_llama_instance

        interpreter = UniversalInterpreter(model_path=model_path, load_llm=True)
        assert interpreter.llm is not None
        mock_llama_class.assert_called_once()


def test_interpret_with_llm_success(monkeypatch):
    """Test successful interpretation with LLM."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.interpret("A cat is an animal")
    assert result["intent"] == "statement_of_fact"
    assert "cat" in result["key_topics"]
    assert result["relation"]["subject"] == "cat"


def test_interpret_with_llm_failure(monkeypatch):
    """Test interpretation failure with LLM."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    result = interpreter.interpret("test input")
    assert result["intent"] == "unknown"
    assert "Could not fully interpret" in result["full_text_rephrased"]


def test_interpret_cache_hit(monkeypatch):
    """Test interpretation cache hit."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.interpretation_cache["test input"] = {
        "intent": "greeting",
        "entities": [],
        "relation": None,
        "key_topics": ["test"],
        "full_text_rephrased": "Hello test",
    }

    result = interpreter.interpret("test input")
    assert result["intent"] == "greeting"


def test_interpret_json_parsing_error(monkeypatch):
    """Test interpretation with invalid JSON response."""

    class MockBadJsonLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": "invalid json"}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockBadJsonLlama()

    result = interpreter.interpret("test input")
    assert result["intent"] == "unknown"


def test_interpret_negation_detection(monkeypatch):
    """Test negation detection in relations."""

    class MockNegationLlama:
        def __call__(self, *args, **kwargs):
            return {
                "choices": [
                    {
                        "text": '{"intent": "statement_of_fact", "entities": [], "relation": {"subject": "cat", "verb": "is not", "object": "dog"}, "key_topics": ["cat"], "full_text_rephrased": "A cat is not a dog."}'
                    }
                ]
            }

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockNegationLlama()

    result = interpreter.interpret("A cat is not a dog")
    assert result["relation"]["properties"]["negated"] is True
    assert result["relation"]["verb"] == "is"
    assert result["relation"]["object"] == "dog"


def test_resolve_context_with_llm(monkeypatch):
    """Test context resolution with LLM."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    history = ["User: What is an apple?"]
    result = interpreter.resolve_context(history, "what color is it?")
    assert "apple" in result


def test_resolve_context_no_change(monkeypatch):
    """Test context resolution when no change needed."""

    class MockNoChangeLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": "what color is the sky?"}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockNoChangeLlama()

    history = ["User: What is the sky?"]
    result = interpreter.resolve_context(history, "what color is the sky?")
    assert result == "what color is the sky?"


def test_resolve_context_llm_error(monkeypatch):
    """Test context resolution with LLM error."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    history = ["User: What is an apple?"]
    result = interpreter.resolve_context(history, "what color is it?")
    assert result == "what color is it?"


def test_decompose_sentence_to_relations_success(monkeypatch):
    """Test successful sentence decomposition."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.decompose_sentence_to_relations("A cat is an animal", "cat")
    assert len(result) == 1
    assert result[0]["subject"] == "cat"
    assert result[0]["verb"] == "is_a"
    assert result[0]["object"] == "animal"


def test_decompose_sentence_to_relations_no_brackets(monkeypatch):
    """Test decomposition with no JSON brackets in response."""

    class MockNoBracketsLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": "no json here"}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockNoBracketsLlama()

    result = interpreter.decompose_sentence_to_relations("test", "topic")
    assert result == []


def test_decompose_sentence_to_relations_invalid_json(monkeypatch):
    """Test decomposition with invalid JSON."""

    class MockInvalidJsonLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": "[invalid json"}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockInvalidJsonLlama()

    result = interpreter.decompose_sentence_to_relations("test", "topic")
    assert result == []


def test_decompose_sentence_to_relations_not_list(monkeypatch):
    """Test decomposition when response is not a list."""

    class MockNotListLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": '{"not": "a list"}'}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockNotListLlama()

    result = interpreter.decompose_sentence_to_relations("test", "topic")
    assert result == []


def test_break_down_definition_success(monkeypatch):
    """Test successful definition breakdown."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.break_down_definition("cat", "a domestic animal")
    assert len(result) == 2
    assert "Cat is a mammal" in result


def test_break_down_definition_empty_response(monkeypatch):
    """Test definition breakdown with empty response."""

    class MockEmptyLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": ""}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockEmptyLlama()

    result = interpreter.break_down_definition("cat", "a domestic animal")
    assert result == []


def test_break_down_definition_llm_error(monkeypatch):
    """Test definition breakdown with LLM error."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    result = interpreter.break_down_definition("cat", "a domestic animal")
    assert result == []


def test_generate_curious_questions_success(monkeypatch):
    """Test successful question generation."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.generate_curious_questions("cat", "a cat is a pet")
    assert len(result) == 2
    assert "What do cats eat?" in result


def test_generate_curious_questions_empty_response(monkeypatch):
    """Test question generation with empty response."""

    class MockEmptyLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": ""}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockEmptyLlama()

    result = interpreter.generate_curious_questions("cat", "a cat is a pet")
    assert result == []


def test_generate_curious_questions_llm_error(monkeypatch):
    """Test question generation with LLM error."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    result = interpreter.generate_curious_questions("cat", "a cat is a pet")
    assert result == []


def test_verify_and_reframe_fact_success(monkeypatch):
    """Test successful fact verification and reframing."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.verify_and_reframe_fact("cat", "A cat is a domestic animal")
    assert result == "Cat is a domestic animal"


def test_verify_and_reframe_fact_rejected(monkeypatch):
    """Test fact verification when fact is rejected."""

    class MockRejectLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": "None"}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockRejectLlama()

    result = interpreter.verify_and_reframe_fact("cat", "Bitcoin is a cryptocurrency")
    assert result is None


def test_verify_and_reframe_fact_llm_error(monkeypatch):
    """Test fact verification with LLM error."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    result = interpreter.verify_and_reframe_fact("cat", "A cat is a domestic animal")
    assert result is None


def test_verify_and_reframe_fact_topic_match(monkeypatch):
    """Test fact verification with topic match when LLM disabled."""
    interpreter = UniversalInterpreter(load_llm=False)

    result = interpreter.verify_and_reframe_fact("cat", "A cat is a domestic animal")
    assert result == "A cat is a domestic animal"


def test_verify_and_reframe_fact_no_topic_match(monkeypatch):
    """Test fact verification with no topic match when LLM disabled."""
    interpreter = UniversalInterpreter(load_llm=False)

    result = interpreter.verify_and_reframe_fact("cat", "A dog is a domestic animal")
    assert result is None


def test_synthesize_with_llm_success(monkeypatch):
    """Test successful synthesis with LLM."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.synthesize("a cat is an animal")
    assert result == "A cat is an animal"


def test_synthesize_with_question(monkeypatch):
    """Test synthesis with original question."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.synthesize("facts", original_question="What is a cat?")
    assert "A cat is an animal" in result


def test_synthesize_cache_hit(monkeypatch):
    """Test synthesis cache hit."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.synthesis_cache["statement|None|facts"] = "cached result"

    result = interpreter.synthesize("facts")
    assert result == "cached result"


def test_synthesize_with_list_input(monkeypatch):
    """Test synthesis with list input."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.synthesize(["fact1", "fact2"])
    assert "A cat is an animal" in result


def test_generate_curriculum_success(monkeypatch):
    """Test successful curriculum generation."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.generate_curriculum("Learn about cats")
    assert len(result) == 5
    assert "mammal" in result


def test_generate_curriculum_empty_response(monkeypatch):
    """Test curriculum generation with empty response."""

    class MockEmptyLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": ""}]}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockEmptyLlama()

    result = interpreter.generate_curriculum("Learn about cats")
    assert result == []


def test_generate_curriculum_llm_error(monkeypatch):
    """Test curriculum generation with LLM error."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    result = interpreter.generate_curriculum("Learn about cats")
    assert result == []


def test_is_pronoun_present():
    """Test pronoun detection."""
    interpreter = UniversalInterpreter(load_llm=False)

    assert interpreter._is_pronoun_present("what is it?") is True
    assert interpreter._is_pronoun_present("what is a cat?") is False
    assert interpreter._is_pronoun_present("they are animals") is True
    assert interpreter._is_pronoun_present("the sky is blue") is False


def test_load_cache_file_not_exists():
    """Test cache loading when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "nonexistent.json"
        interpreter = UniversalInterpreter(load_llm=False, cache_file=cache_file)
        assert len(interpreter.interpretation_cache) == 0
        assert len(interpreter.synthesis_cache) == 0


def test_load_cache_invalid_json(tmp_path):
    """Test cache loading with invalid JSON."""
    cache_file = tmp_path / "invalid.json"
    cache_file.write_text("{invalid json}")

    interpreter = UniversalInterpreter(load_llm=False, cache_file=cache_file)
    assert len(interpreter.interpretation_cache) == 0
    assert len(interpreter.synthesis_cache) == 0


def test_save_cache_error(tmp_path):
    """Test cache saving with error."""
    # Create a directory with the same name as the cache file to cause an error
    cache_file = tmp_path / "cache.json"
    cache_file.mkdir()

    interpreter = UniversalInterpreter(load_llm=False, cache_file=cache_file)
    interpreter.synthesis_cache["test"] = "value"
    interpreter._save_cache()  # Should not raise an exception


def test_run_llm_completion_with_retries_success(monkeypatch):
    """Test successful LLM completion with retries."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter._run_llm_completion_with_retries("test prompt", 100)
    assert result != ""


def test_run_llm_completion_with_retries_no_llm(monkeypatch):
    """Test LLM completion when LLM is None."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = None

    result = interpreter._run_llm_completion_with_retries("test prompt", 100)
    assert result == ""


def test_run_llm_completion_with_retries_invalid_response(monkeypatch):
    """Test LLM completion with invalid response format."""

    class MockInvalidResponseLlama:
        def __call__(self, *args, **kwargs):
            return {"invalid": "response"}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockInvalidResponseLlama()

    result = interpreter._run_llm_completion_with_retries("test prompt", 100)
    assert result == ""


def test_run_llm_completion_with_retries_empty_choices(monkeypatch):
    """Test LLM completion with empty choices."""

    class MockEmptyChoicesLlama:
        def __call__(self, *args, **kwargs):
            return {"choices": []}

    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockEmptyChoicesLlama()

    result = interpreter._run_llm_completion_with_retries("test prompt", 100)
    assert result == ""


def test_run_llm_completion_with_retries_all_fail(monkeypatch):
    """Test LLM completion when all retries fail."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockFailingLlama()

    result = interpreter._run_llm_completion_with_retries("test prompt", 100)
    assert result == ""


def test_interpret_with_context_no_pronouns():
    """Test interpret_with_context when no pronouns present."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.interpret_with_context(
        "what is a cat?", ["User: tell me about animals"]
    )
    assert result["intent"] == "statement_of_fact"


def test_interpret_with_context_empty_history():
    """Test interpret_with_context with empty history."""
    interpreter = UniversalInterpreter(load_llm=False)
    interpreter.llm = MockWorkingLlama()

    result = interpreter.interpret_with_context("what is a cat?", [])
    assert result["intent"] == "statement_of_fact"
