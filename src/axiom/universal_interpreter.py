from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
    cast,
)

from llama_cpp import Llama

from .config import DEFAULT_CACHE_FILE, DEFAULT_LLM_PATH

logger = logging.getLogger(__name__)

LLM_INTERACTIONS_LOG = Path("llm_interactions.jsonl")


def log_llm_interaction(prompt: str, ideal_response: str) -> None:
    """Appends a prompt and its ideal response to the dataset log file."""
    interaction = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ideal_response},
        ]
    }
    with LLM_INTERACTIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(interaction) + "\n")


REPHRASING_PROMPT: Final = """
You are a language rephrasing engine. Your task is to convert the given 'Facts' into a single, natural English sentence. You are a fluent parrot. You must follow these rules strictly:
1.  **ONLY use the information given in the 'Facts' string.**
2.  **DO NOT add any extra information, commentary, or meta-analysis.**
3.  **DO NOT apologize or mention your own limitations.**
4.  **Your output must be ONLY the rephrased sentence and nothing else.**
"""[1:-1]

JSON_STRUCTURE_PROMPT: Final = """
The JSON object must have the following fields:
- 'intent': Classify the user's primary intent. Possible values are: 'greeting', 'farewell', 'question_about_entity', 'question_about_concept', 'statement_of_fact', 'statement_of_correction', 'gratitude', 'acknowledgment', 'positive_affirmation', 'command', 'unknown'.
- 'relation': If 'statement_of_fact' or 'statement_of_correction', extract the core relationship. This object has fields: 'subject', 'verb', 'object', and optional 'predicate', 'relation', or 'properties'.
- 'key_topics': A list of the main subjects or topics.
- 'full_text_rephrased': A neutral, one-sentence rephrasing.
"""[1:-1]


Intent: TypeAlias = Literal[
    "greeting",
    "farewell",
    "question_about_entity",
    "question_about_concept",
    "statement_of_fact",
    "statement_of_correction",
    "gratitude",
    "acknowledgment",
    "positive_affirmation",
    "command",
    "unknown",
    "unknown_verb_failure",
    "question_yes_no",
    "meta_question_self",
    "meta_question_purpose",
    "meta_question_abilities",
    "command_show_all_facts",
    "question_by_relation",
]


class PropertyData(TypedDict, total=False):
    """Optional metadata about a relationship, such as time or location."""

    effective_date: NotRequired[str]
    location: NotRequired[str]
    confidence: NotRequired[float]
    provenance: NotRequired[str]
    negated: NotRequired[bool]
    contradicted: NotRequired[bool]
    revision_status: NotRequired[
        Literal[
            "active",
            "superseded",
            "disputed",
            "replaced",
            "ignored_lower_provenance",
            "merged",
        ]
    ]
    superseded_by: NotRequired[str]
    last_modified: NotRequired[float]
    confidence_updated: NotRequired[bool]


class RelationData(TypedDict, total=False):
    """Defines a structured relationship extracted from a sentence."""

    subject: str | list[Any] | dict[str, Any]
    verb: NotRequired[str]
    object: str | list[Any] | dict[str, Any]
    predicate: NotRequired[str]
    relation: NotRequired[str]
    properties: NotRequired[PropertyData]


class Entity(TypedDict):
    """Represents a key concept or entity extracted from text."""

    name: str
    type: Literal["CONCEPT", "PERSON", "ROLE", "PROPERTY"]


class InterpretData(TypedDict):
    """Structured result from the interpretation step."""

    intent: Intent
    entities: list[Entity]
    relation: RelationData | None
    key_topics: list[str]
    full_text_rephrased: str
    provenance: NotRequired[str]
    confidence: NotRequired[float]


class Task(TypedDict):
    """A structured representation of a learning task."""

    task: Literal["INVESTIGATE", "CLARIFY"]
    topic: str
    context: NotRequired[str]


PRONOUNS: Final = ("it", "its", "they", "them", "their", "he", "she", "his", "her")


class UniversalInterpreter:
    """Provides an LLM-based interface for complex language tasks.

    This class acts as the agent's fallback "senses" for understanding
    language that is too complex for the `SymbolicParser`. It wraps a
    local LLM (via `llama-cpp-python`) and uses carefully crafted prompts
    to perform structured interpretation, context resolution, and natural
    language synthesis.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_LLM_PATH,
        cache_file: str | Path = DEFAULT_CACHE_FILE,
        load_llm: bool = True,
    ) -> None:
        """Initialize the UniversalInterpreter and load the LLM into memory."""
        logger.info(
            "[yellow]Initializing Universal Interpreter (loading Mini LLM if enabled)...[/yellow]"
        )
        self.llm: Llama | None = None

        if load_llm:
            logger.info(
                "[bold cyan]   - Attempting to load LLM from path: %s[/bold cyan]",
                model_path,
            )

            if not Path(model_path).exists():
                logger.error(
                    "[bold red]   - ❌ LLM model file not found at path: %s[/bold red]",
                    model_path,
                )
                raise FileNotFoundError(f"Interpreter model not found at {model_path}.")

            with open(os.devnull, "w") as f, redirect_stderr(f):
                self.llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=0,
                    n_ctx=16384,
                    n_threads=os.cpu_count(),
                    n_batch=1024,
                    verbose=False,
                )
            logger.info(
                "[bold green]   - ✅ Successfully loaded LLM: %s[/bold green]",
                Path(model_path).name,
            )
        else:
            logger.info("Initializing Universal Interpreter in SYMBOLIC-ONLY mode.")

        self.interpretation_cache: dict[str, InterpretData] = {}
        self.synthesis_cache: dict[str, str] = {}
        self.cache_file = Path(cache_file)
        self._load_cache()

        logger.info("[success]Universal Interpreter loaded successfully.[/success]")

    def _is_pronoun_present(self, text: str) -> bool:
        """Check if any pronoun exists as a whole word in the text."""
        for pronoun in PRONOUNS:
            if re.search(rf"\b{pronoun}\b", text, re.IGNORECASE):
                return True
        return False

    def _load_cache(self) -> None:
        """Load the interpretation and synthesis caches from a JSON file."""
        if not self.cache_file.exists():
            logger.info("[Cache]: No cache file found. Starting fresh.")
            return

        try:
            with self.cache_file.open("rb") as fp:
                cache_data = json.load(fp)
                self.interpretation_cache = dict(
                    cache_data.get("interpretations", []),
                )
                self.synthesis_cache = dict(cache_data.get("synthesis", []))
            logger.info(
                "[border][Cache]: Loaded %d interpretation(s) and %d synthesis caches from %s.[/border]",
                len(self.interpretation_cache),
                len(self.synthesis_cache),
                self.cache_file,
            )
        except Exception as exc:
            logger.warning(
                "[Cache Error]: Could not load cache file. Starting fresh. Error: %s",
                exc,
            )

    def _save_cache(self) -> None:
        """Save the current interpretation and synthesis caches to a JSON file."""
        try:
            with self.cache_file.open("w", encoding="utf-8") as f:
                cache_data = {
                    "interpretations": list(self.interpretation_cache.items()),
                    "synthesis": list(self.synthesis_cache.items()),
                }
                json.dump(cache_data, f, indent=4)
        except Exception as exc:
            logger.error(
                "[Cache Error]: Could not save cache to %s. Error: %s",
                self.cache_file,
                exc,
            )

    def get_prompt_for_test(self, task_name: str, **kwargs: Any) -> str:
        """
        A helper method for the data generation script. It returns the exact
        prompt that would be sent to the LLM for a given task and inputs.
        """
        if task_name == "interpret":
            system_prompt = "You are a strict, precise factual analysis engine..."
            examples_list = [
                "Input: 'show all facts'\nOutput: {...}",
            ]
            examples_prompt = "Here are some examples:\n" + "\n\n".join(examples_list)
            sanitized_input = json.dumps(kwargs.get("user_input", ""))
            return f"[INST] {system_prompt}\n\n{JSON_STRUCTURE_PROMPT}\n\n{examples_prompt}\n\nNow, analyze the following user input and provide ONLY the JSON output:\n{sanitized_input}[/INST]"

        if task_name == "decomposition":
            text = kwargs.get("text", "")
            main_topic = kwargs.get("main_topic")
            topic_context = (
                f"The primary topic of this sentence is '{main_topic}'. "
                "Ensure the subject of at least one core relation is this topic or a direct synonym."
                if main_topic
                else ""
            )
            return f"""
            **ROLE:** You are a knowledge engineering system...
            **CONTEXT:** {topic_context}
            **SENTENCE TO ANALYZE:**
            "{text}"
            ...
            """

        if task_name == "synthesis_statement":
            facts_str = json.dumps(kwargs.get("facts_to_rephrase", []))
            return (
                f"[INST] {REPHRASING_PROMPT}\n\nFacts to rephrase: '{facts_str}'[/INST]"
            )

        if task_name == "question_generation":
            topic = kwargs.get("topic", "")
            known_fact = kwargs.get("known_fact", "")
            system_prompt = "You are a creative, inquisitive assistant..."
            examples_prompt = "Here are some examples:\n..."
            return f"[INST] {system_prompt}\n\n{examples_prompt}\n\nTopic: {topic}\nKnown Fact: {known_fact}\nOutput:[/INST]"

        return ""

    def _clean_llm_json_output(self, raw_text: str) -> str:
        """Clean and extract a JSON object from the raw output of an LLM."""
        start_brace = raw_text.find("{")
        end_brace = raw_text.rfind("}")
        if start_brace == -1 or end_brace == -1:
            start_bracket = raw_text.find("[")
            end_bracket = raw_text.rfind("]")
            if start_bracket != -1 and end_bracket != -1:
                json_str = raw_text[start_bracket : end_bracket + 1]
                return re.sub(r",\s*(\}|\])", r"\1", json_str).replace("\n", "")
            return ""
        json_str = raw_text[start_brace : end_brace + 1]
        return re.sub(r",\s*(\}|\])", r"\1", json_str).replace("\n", "")

    def _run_llm_completion_with_retries(
        self, prompt: str, max_tokens: int, temperature: float = 0.0
    ) -> str:
        """Runs the LLM completion with retry and backoff logic."""
        retries = 3
        backoff_factor = 2

        for attempt in range(retries):
            try:
                if not self.llm:
                    raise ConnectionError("LLM is not loaded.")

                output = cast(
                    "dict[str, list[dict[str, str]]]",
                    self.llm(
                        prompt,
                        max_tokens=max_tokens,
                        stop=["</s>", "\n\n"],
                        echo=False,
                        temperature=temperature,
                    ),
                )

                if output and "choices" in output and output["choices"]:
                    return output["choices"][0]["text"].strip()

                raise ValueError("Invalid LLM response format.")

            except (ConnectionError, ValueError, IndexError) as e:
                logger.warning(
                    "  [LLM Error]: Attempt %d/%d failed: %s", attempt + 1, retries, e
                )
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2**attempt)
                    logger.info("  [LLM Retry]: Retrying in %d seconds...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error("  [LLM Error]: All retry attempts failed.")
                    return ""
        return ""

    def interpret(self, user_input: str) -> InterpretData:
        """Analyze user input with the LLM and return a structured interpretation."""
        cache_key = user_input
        if cache_key in self.interpretation_cache:
            logger.info("  [Interpreter Cache]: Hit!")
            return self.interpretation_cache[cache_key]

        if self.llm is None:
            logger.warning(
                "  [Interpreter Error]: LLM is disabled. Cannot interpret complex input."
            )
            return cast(
                "InterpretData",
                {
                    "intent": "unknown",
                    "entities": [],
                    "relation": None,
                    "key_topics": user_input.split(),
                    "full_text_rephrased": f"Could not interpret (LLM disabled): '{user_input}'",
                    "provenance": "llm",
                    "confidence": 0.0,
                },
            )

        logger.info("  [Interpreter Cache]: Miss. Running LLM for interpretation.")

        system_prompt = (
            "You are a strict, precise factual analysis engine. Your PRIMARY task is to identify the user's intent. "
            "If the input is a declarative sentence that presents a fact, you MUST classify the intent as 'statement_of_fact'. "
            "Your secondary task is to extract the relationship into a structured JSON object. "
            "Your output must be a single, valid JSON object and NOTHING else."
        )
        json_structure_prompt = JSON_STRUCTURE_PROMPT

        examples_list = [
            'Input: \'show all facts\'\nOutput: {"intent": "command", "entities": [], "relation": null, "key_topics": ["show all facts"], "full_text_rephrased": "User has issued a command to show all facts."}',
            'Input: \'what is a human\'\nOutput: {"intent": "question_about_concept", "entities": [{"name": "human", "type": "CONCEPT"}], "relation": null, "key_topics": ["human"], "full_text_rephrased": "User is asking for information about a human."}',
            'Input: \'correction: the sky is blue\'\nOutput: {"intent": "statement_of_correction", "entities": [{"name": "sky", "type": "CONCEPT"}, {"name": "blue", "type": "COLOR"}], "relation": {"subject": "the sky", "verb": "is", "object": "blue"}, "key_topics": ["sky", "blue"], "full_text_rephrased": "User is correcting the fact about the sky to state that it is blue."}',
            'Input: \'In 2023, Tim Cook was the CEO of Apple.\'\nOutput: {"intent": "statement_of_fact", "entities": [{"name": "Tim Cook", "type": "PERSON"}, {"name": "CEO of Apple", "type": "ROLE"}], "relation": {"subject": "Tim Cook", "verb": "was", "object": "the CEO of Apple", "properties": {"effective_date": "2023-01-01"}}, "key_topics": ["Tim Cook", "Apple", "CEO"], "full_text_rephrased": "User is stating that Tim Cook was the CEO of Apple in 2023."}',
            'Input: \'who is Donald Trump?\'\nOutput: {"intent": "question_about_entity", "entities": [{"name": "Donald Trump", "type": "PERSON"}], "relation": null, "key_topics": ["Donald Trump"], "full_text_rephrased": "User is asking for information about Donald Trump."}',
        ]
        examples_prompt = "Here are some examples:\n" + "\n\n".join(examples_list)

        sanitized_input = json.dumps(user_input)
        full_prompt = (
            f"[INST] {system_prompt}\n\n{json_structure_prompt}\n\n{examples_prompt}\n\n"
            f"Now, analyze the following user input and provide ONLY the JSON output:\n{sanitized_input}[/INST]"
        )
        try:
            response_text = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=512
            )
            cleaned_json_str = self._clean_llm_json_output(response_text)
            if not cleaned_json_str:
                raise json.JSONDecodeError("No JSON object found", response_text, 0)
            raw_interpretation = json.loads(cleaned_json_str)

            interpretation = cast("InterpretData", raw_interpretation)

            if isinstance(interpretation, dict):
                interpretation.setdefault("provenance", "llm")
                interpretation.setdefault("confidence", 0.6)
                interpretation.setdefault("key_topics", [])
                interpretation.setdefault("full_text_rephrased", "")

                rel = interpretation.get("relation")
                if isinstance(rel, dict):
                    props = rel.setdefault("properties", cast("PropertyData", {}))
                    props.setdefault("confidence", 0.6)
                    props.setdefault("provenance", "llm")

                    raw_verb = rel.get("verb", "").lower()
                    raw_object_text = str(rel.get("object", ""))

                    if re.search(
                        r"\b(not|never|no|without)\b",
                        raw_verb,
                        re.IGNORECASE,
                    ) or re.search(
                        r"\b(not|never|no|without)\b",
                        raw_object_text,
                        re.IGNORECASE,
                    ):
                        props["negated"] = True

                        rel["object"] = re.sub(
                            r"\b(not|never|no|without)\b",
                            "",
                            raw_object_text,
                            flags=re.IGNORECASE,
                        ).strip()
                        rel["verb"] = re.sub(
                            r"\b(not|never|no|without)\b",
                            "",
                            raw_verb,
                            flags=re.IGNORECASE,
                        ).strip()

                    interpretation["confidence"] = props.get("confidence", 0.6)

            self.interpretation_cache[cache_key] = interpretation
            self._save_cache()
            return interpretation
        except Exception as e:
            logger.error(
                "  [Interpreter Error]: Could not parse LLM output. Error: %s", e
            )
            return cast(
                "InterpretData",
                {
                    "intent": "unknown",
                    "entities": [],
                    "relation": None,
                    "key_topics": user_input.split(),
                    "full_text_rephrased": f"Could not fully interpret: '{user_input}'",
                    "provenance": "llm",
                    "confidence": 0.0,
                },
            )

    def resolve_context(self, history: list[str], new_input: str) -> str:
        """Use the LLM to perform coreference resolution on the user's input."""
        if self.llm is None:
            logger.warning(
                "  [Context Resolver Error]: LLM is disabled. Cannot resolve context."
            )
            return new_input

        logger.info("  [Context Resolver]: Attempting to resolve pronouns...")
        formatted_history = "\n".join(history)
        system_prompt = (
            "You are a strict coreference resolution engine. Your one and only task is to rephrase the 'New Input' "
            "by replacing pronouns (like it, its, they, them, their) with the specific noun they refer to from the 'Conversation History'.\n"
            "RULES:\n"
            "1. You MUST replace the pronoun with the full noun phrase from the history.\n"
            "2. If the New Input contains NO pronouns, you MUST return it completely unchanged.\n"
            "3. Your output MUST be ONLY the rephrased sentence and nothing else."
        )
        examples_prompt = (
            "Here are some examples:\n"
            "Conversation History:\n"
            "User: what is an apple?\nAgent: An apple is a fruit.\n"
            "New Input: what color is it?\nOutput: what color is an apple?\n\n"
            "Conversation History:\n"
            "User: tell me about dogs\nAgent: Dogs are mammals.\n"
            "New Input: what do they eat?\nOutput: what do dogs eat?\n\n"
            "Conversation History:\n"
            "User: tell me about the solar system\nAgent: The solar system has eight planets.\n"
            "New Input: what is the largest planet?\nOutput: what is the largest planet?"
        )
        full_prompt = (
            f"[INST] {system_prompt}\n\n{examples_prompt}\n\n"
            f"Conversation History:\n{formatted_history}\n"
            f"New Input: {new_input}\nOutput:[/INST]"
        )
        try:
            rephrased_input = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=128
            )
            if not rephrased_input:
                raise ValueError(
                    "LLM returned an empty response for context resolution."
                )
            if rephrased_input.lower() != new_input.lower():
                logger.info(
                    "    - Context resolved: '%s' -> '%s'", new_input, rephrased_input
                )
                return rephrased_input
            logger.info("    - No context to resolve, using original input.")
            return new_input
        except Exception as e:
            logger.error(
                "  [Context Resolver Error]: Could not resolve context. Error: %s", e
            )
            return new_input

    def interpret_with_context(
        self,
        user_input: str,
        history: list[str],
    ) -> InterpretData:
        """Interpret user input after first attempting to resolve context."""
        contextual_input = user_input

        if history and self._is_pronoun_present(user_input):
            contextual_input = self.resolve_context(history, user_input)

        return self.interpret(contextual_input)

    def _flatten_structured_object(self, obj: Any) -> str:
        """
        Recursively traverses a potentially nested dictionary or list from the LLM
        and extracts all meaningful string values into a single, clean phrase.
        """
        if isinstance(obj, str):
            return obj.strip()

        parts = []
        if isinstance(obj, dict):
            if "value" in obj and isinstance(obj["value"], str):
                return obj["value"].strip()
            if "name" in obj and isinstance(obj["name"], str):
                return obj["name"].strip()

            for key, value in obj.items():
                if key.lower() != "type":
                    parts.append(self._flatten_structured_object(value))

        elif isinstance(obj, list):
            parts = [self._flatten_structured_object(item) for item in obj]

        return " ".join(filter(None, parts)).strip()

    def decompose_sentence_to_relations(
        self, text: str, main_topic: str | None = None
    ) -> list[RelationData]:
        """
        Uses the LLM to decompose a sentence into a list of atomic relations,
        now with robust validation and input sanitization.
        """
        caller_name = f"{self.__class__.__name__}.decompose_sentence_to_relations"
        if self.llm is None:
            logger.warning("[Interpreter]: LLM is disabled. Cannot decompose text.")
            return []

        logger.info(
            "[purple]   - [Interpreter]: Decomposing sentence into atomic facts...[/purple]"
        )

        sanitized_text = text.replace("\\", "\\\\").replace('"', '\\"')

        topic_context = (
            f"The primary topic of this sentence is '{main_topic}'. "
            "Ensure the subject of at least one core relation is this topic or a direct synonym."
            if main_topic
            else ""
        )

        prompt = f"""
        **ROLE:** You are a silent, non-conversational knowledge engineering system.

        **TASK:** Analyze the user's sentence and break it down into a list of simple, atomic semantic relations.
        - Each relation MUST be a JSON object: {{"subject": "...", "verb": "...", "object": "..."}}
        - Subjects and objects should be simple concepts.
        - Verbs should be concise, standardized predicates (e.g., "is_a", "has_property").

        **CONTEXT:** {topic_context}

        **EXAMPLE:**
        Sentence: "Photosynthesis is a process used by plants to convert light energy into chemical energy."
        Output:
        [
          {{"subject": "photosynthesis", "verb": "is_a", "object": "process"}},
          {{"subject": "photosynthesis", "verb": "is_used_by", "object": "plants"}}
        ]

        **SENTENCE TO ANALYZE:**
        "{sanitized_text}"

        **RULES (MANDATORY):**
        1.  Your entire response MUST be ONLY the JSON list.
        2.  Do NOT include markdown, explanations, apologies, or conversational filler like "Here is the JSON...".
        3.  If no facts can be extracted, return an empty list `[]`.
        """
        try:
            response_text = self._run_llm_completion_with_retries(
                f"[INST]{prompt}[/INST]", max_tokens=1024, temperature=0.1
            )
            if not response_text:
                logger.warning(
                    "  [Interpreter Warning]: LLM returned an empty response for decomposition."
                )
                return []

            start_bracket = response_text.find("[")
            end_bracket = response_text.rfind("]")
            if start_bracket == -1 or end_bracket == -1:
                logger.warning(
                    f"  [Interpreter Warning]: LLM response did not contain a JSON list. Output: {response_text}"
                )
                return []

            json_str = response_text[start_bracket : end_bracket + 1]
            try:
                relations_raw = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(
                    f"  [Interpreter Error]: Failed to decompose sentence. Error: {e}"
                )
                logger.debug(
                    f"  [Interpreter Debug]: Malformed JSON from LLM:\n{json_str}"
                )
                return []

            if not isinstance(relations_raw, list):
                logger.warning(
                    f"  [Interpreter Warning]: LLM returned valid JSON, but it was not a list. Output: {json_str}"
                )
                return []

            validated_relations: list[RelationData] = []
            for relation in relations_raw:
                if not isinstance(relation, dict):
                    continue

                subject = relation.get("subject")
                verb = relation.get("verb")
                obj = relation.get("object")

                if not isinstance(subject, str):
                    subject = " ".join(self._flatten_structured_object(subject).split())
                    relation["subject"] = subject

                if not isinstance(obj, str):
                    obj = " ".join(self._flatten_structured_object(obj).split())
                    relation["object"] = obj

                if (
                    isinstance(subject, str)
                    and isinstance(verb, str)
                    and isinstance(obj, str)
                ):
                    validated_relations.append(cast("RelationData", relation))
                else:
                    logger.warning(
                        f"  [Interpreter Validation]: Discarding malformed relation after flattening: {relation} (in {caller_name})"
                    )

            if validated_relations:
                logger.info(
                    f"[green]   - Decomposed and validated {len(validated_relations)} atomic relations.[/green]"
                )
                return validated_relations

            logger.warning(
                f"  [Interpreter Warning]: No valid atomic relations could be extracted after validation. (in {caller_name})"
            )
            return []

        except Exception as e:
            logger.error(
                f"  [Interpreter Error]: An unexpected error occurred during decomposition. Error: {e} (in {caller_name})"
            )
            return []

    def break_down_definition(self, subject: str, chunky_definition: str) -> list[str]:
        """Use the LLM to break a complex definition into simple, atomic facts."""
        caller_name = f"{self.__class__.__name__}.break_down_definition"
        if self.llm is None:
            logger.warning(
                "  [Interpreter Error]: LLM is disabled. Cannot break down definition."
            )
            return []

        logger.info(
            "  [Interpreter]: Breaking down chunky definition for '%s'...", subject
        )
        system_prompt = (
            "You are a logical decomposition engine. Your task is to break down a "
            "complex 'Definition' about a 'Subject' into a list of simple, atomic, "
            "declarative sentences. Each sentence must be a standalone fact.\n"
            "RULES:\n"
            "1. Each output sentence MUST start with the original 'Subject'.\n"
            "2. The output MUST be a simple list, with each sentence on a new line, "
            "prefixed with a hyphen.\n"
            "3. DO NOT add any other text, explanation, or commentary."
        )
        examples_prompt = (
            "Here are some examples:\n"
            "Subject: Bacteria\n"
            "Definition: ubiquitous, mostly free-living organisms often consisting of one biological cell\n"
            "Output:\n"
            "- Bacteria are ubiquitous.\n"
            "- Bacteria are mostly free-living organisms.\n"
            "- Bacteria consist of one biological cell.\n\n"
            "Subject: SymbolicParser\n"
            "Definition: a deterministic, rule-based parser for understanding simple language\n"
            "Output:\n"
            "- SymbolicParser is a deterministic parser.\n"
            "- SymbolicParser is a rule-based parser.\n"
            "- SymbolicParser is for understanding simple language."
        )
        full_prompt = (
            f"[INST] {system_prompt}\n\n{examples_prompt}\n\n"
            f"Subject: {subject.capitalize()}\nDefinition: {chunky_definition}\n"
            f"Output:[/INST]"
        )
        try:
            response_text = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=256, temperature=0.2
            )
            atomic_sentences = [
                s.strip()
                for s in response_text.replace("-", "").split("\n")
                if s.strip()
            ]

            if atomic_sentences:
                logger.info(
                    "    - Decomposed into %d atomic facts.", len(atomic_sentences)
                )
                return atomic_sentences
            return []
        except Exception as e:
            logger.error(
                "  [Interpreter Error]: Could not break down definition. Error: %s",
                e,
                caller_name,
            )
            return []

    def generate_curious_questions(self, topic: str, known_fact: str) -> list[str]:
        """Generate simple, fundamental follow-up questions about a topic."""
        caller_name = f"{self.__class__.__name__}.generate_curious_questions"
        if self.llm is None:
            logger.warning(
                "  [Question Generation Error]: LLM is disabled. Cannot generate questions."
            )
            return []

        system_prompt = (
            "You are a creative, inquisitive assistant that thinks like a curious child. "
            "Your task is to generate exactly two, simple, fundamental follow-up questions about a topic, "
            "based on a fact that is already known. The questions should be things a child would ask to learn more.\n"
            "RULES:\n"
            "1. Generate exactly TWO questions.\n"
            "2. The questions must be simple and fundamental.\n"
            "3. Format your output as a simple list, with each question on a new line, prefixed with a hyphen.\n"
            "4. DO NOT add any other text, explanation, or commentary."
        )
        examples_prompt = (
            "Here are some examples:\n"
            "Topic: Apple\nKnown Fact: An apple is a fruit.\n"
            "Output:\n- Where do apples grow?\n- What is inside an apple?\n\n"
            "Topic: Sun\nKnown Fact: The sun is a star.\n"
            "Output:\n- Why is the sun hot?\n- How big is the sun?"
        )
        full_prompt = (
            f"[INST] {system_prompt}\n\n{examples_prompt}\n\n"
            f"Topic: {topic}\nKnown Fact: {known_fact}\n"
            f"Output:[/INST]"
        )
        try:
            response_text = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=128, temperature=0.8
            )
            questions = [
                q.strip()
                for q in response_text.replace("-", "").split("\n")
                if q.strip()
            ]

            if questions:
                logger.info("    - Generated %d curious questions.", len(questions))
                return questions
            return []
        except Exception as exc:
            logger.error(
                "  [Question Generation Error]: Could not generate questions. Error: %s",
                exc,
                caller_name,
            )
            return []

    def verify_and_reframe_fact(
        self,
        original_topic: str,
        raw_sentence: str,
    ) -> str | None:
        """
        Uses the LLM to verify if a raw sentence is relevant to a topic and,
        if so, reframes it into a simple, atomic S-V-O sentence for learning.
        """
        caller_name = f"{self.__class__.__name__}.verify_and_reframe_fact"
        if self.llm is None:
            logger.warning(
                "  [Fact Verifier Error]: LLM is disabled. Cannot verify/reframe."
            )
            if original_topic.lower() in raw_sentence.lower():
                return raw_sentence
            return None

        logger.info(
            f"[purple]   Asking LLM to verify and reframe fact for '{original_topic}'...[/purple]"
        )
        system_prompt = (
            "You are a precise fact verification and reframing engine. Your task is to analyze a 'Raw Sentence' "
            "to see if it is a direct, useful fact about the 'Original Topic'. If it is, you MUST rephrase it into a "
            "single, simple, declarative sentence (Subject-Verb-Object). If it is not relevant, you MUST output ONLY the word 'None'."
        )
        examples_prompt = (
            "Here are some examples:\n"
            "Original Topic: fabric\n"
            "Raw Sentence: A textile is a flexible material made by creating an interlocking network of yarns or threads.\n"
            "Output: A fabric is a flexible material.\n\n"
            "Original Topic: history of bitcoin\n"
            "Raw Sentence: Bitcoin is a cryptocurrency, a digital asset that uses cryptography to control its creation and management.\n"
            "Output: None\n\n"
            "Original Topic: bees\n"
            "Raw Sentence: Bees are flying insects closely related to wasps and ants, known for their role in pollination.\n"
            "Output: Bees are flying insects."
        )
        full_prompt = (
            f"[INST] {system_prompt}\n\n{examples_prompt}\n\n"
            f"Original Topic: {original_topic}\nRaw Sentence: {raw_sentence}\n"
            f"Output:[/INST]"
        )
        try:
            rephrased_fact = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=64
            )

            if not rephrased_fact:
                logger.error(
                    f"  [Fact Verifier Error]: LLM returned an empty response. (in {caller_name})"
                )
                return None

            if "none" in rephrased_fact.lower():
                logger.info(
                    "  [Fact Verifier]: LLM determined the fact was not relevant."
                )
                return None

            return rephrased_fact.strip().strip(".")

        except Exception as e:
            logger.error(
                f"  [Fact Verifier Error]: Could not process fact with LLM. Error: {e} (in {caller_name})"
            )
            return None

    def synthesize(
        self,
        structured_facts: str | list[Any],
        original_question: str | None = None,
        mode: str = "statement",
    ) -> str:
        """Convert a structured, internal representation into natural language."""
        caller_name = f"{self.__class__.__name__}.synthesize"
        facts_str: str
        if isinstance(structured_facts, list):
            try:
                facts_str = json.dumps([str(f) for f in structured_facts])
            except Exception:
                facts_str = str(structured_facts)
        else:
            facts_str = structured_facts

        cache_key = f"{mode}|{original_question}|{facts_str}"
        if cache_key in self.synthesis_cache:
            logger.info("  [Synthesizer Cache]: Hit!")
            return self.synthesis_cache[cache_key]

        if self.llm is None:
            return facts_str

        logger.info(
            "[yellow][Synthesizer Cache]: Miss. Running LLM for synthesis in '%s' mode.[/yellow]",
            mode,
        )

        system_prompt = ""
        task_prompt = ""

        if mode == "clarification_question":
            system_prompt = (
                "You are an inquisitive AI agent. Your task is to ask a clarifying question. "
                "You have been given two conflicting facts. Formulate a single, polite, and simple question. "
                "Do not state the facts directly. Your output must be ONLY the question."
            )
            task_prompt = f"Conflicting Facts: '{facts_str}'"
        else:
            system_prompt = REPHRASING_PROMPT
            task_prompt = f"Facts to rephrase: '{facts_str}'"
            if original_question:
                task_prompt = f"Using ONLY the facts provided, directly answer the question.\nQuestion: '{original_question}'\nFacts: '{facts_str}'"

        full_prompt = f"[INST] {system_prompt}\n\n{task_prompt}[/INST]"
        try:
            temperature = (
                0.7 if mode in ("clarification_question", "creative_writing") else 0.1
            )
            synthesized_text = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=1024, temperature=temperature
            )
            if not synthesized_text:
                raise ValueError("LLM returned an empty string during synthesis.")

            phrases_to_remove = [
                "rephrased sentence:",
                "based on the provided facts,",
                "the rephrased sentence is:",
                "good output:",
                "output:",
            ]
            for phrase in phrases_to_remove:
                if synthesized_text.lower().startswith(phrase):
                    synthesized_text = synthesized_text[len(phrase) :].strip()

            if "(" in synthesized_text:
                synthesized_text = synthesized_text.split("(")[0].strip()

            self.synthesis_cache[cache_key] = synthesized_text
            self._save_cache()

            return synthesized_text
        except Exception as e:
            logger.error(
                "  [Synthesizer Error]: Could not generate fluent text. Error: %s",
                e,
                caller_name,
            )
            return facts_str

    def generate_curriculum(
        self, high_level_goal: str, style: str | None = None
    ) -> list[str]:
        """
        Uses the LLM to break down a high-level learning goal into a list of
        essential, prerequisite topics to study, guided by a pedagogical style.
        """
        if self.llm is None:
            return []

        style_instruction = (
            f"Generate the topics from the perspective of: {style}."
            if style
            else "Generate a standard list of topics."
        )

        system_prompt = (
            "You are a curriculum design expert. Your task is to break down a 'High-Level Goal' into a "
            "short, prioritized list of the most fundamental, prerequisite concepts needed to understand it. "
            "These concepts should be simple nouns or short noun phrases."
        )
        examples_prompt = (
            "RULES:\n"
            "1. Output ONLY a comma-separated list of topics.\n"
            "2. Prioritize the most foundational concepts first.\n"
            "3. Do not add numbers, bullets, or any other formatting.\n\n"
            "Example 1:\n"
            "High-Level Goal: Understand photosynthesis\n"
            "Output: plant, cell, sunlight, chlorophyll, water, carbon dioxide, oxygen, glucose"
        )

        full_prompt = (
            f"[INST] {system_prompt}\n\n"
            f"{style_instruction}\n\n"
            f"{examples_prompt}\n\n"
            f"High-Level Goal: {high_level_goal}\n"
            f"Output:[/INST]"
        )
        try:
            response_text = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=2048, temperature=0.5
            )

            topics = [
                topic.strip() for topic in response_text.split(",") if topic.strip()
            ]
            logger.info(
                "[purple]   - Generated curriculum with %d topics.[/purple]",
                len(topics),
            )
            return topics
        except Exception as e:
            logger.error(
                "  [Interpreter Error]: Could not generate curriculum. Error: %s",
                e,
            )
            return []

    def generate_new_creative_seed(self) -> str:
        """
        Calls an external LLM to generate a brand new, high-level creative seed
        for curriculum generation when the agent has exhausted its current goals.
        """
        if self.llm is None:
            logger.warning(
                "[Curriculum Generation]: LLM is disabled. Cannot generate a new creative seed."
            )
            return "Investigate the concept of 'knowledge' itself."

        logger.info(
            "[Curriculum Generation]: Current curriculum exhausted. Asking LLM for a new high-level goal..."
        )

        prompt = (
            "You are a curriculum director for an emerging AI. Your task is to devise a single, compelling, high-level "
            "learning objective for the AI. The objective should be a 'creative seed'—a rich, multi-faceted topic that can be "
            "broken down into many stages of learning.\n\n"
            "The theme should be related to the fundamentals of intelligence, consciousness, the universe, or the nature of reality. "
            "Frame the objective as an inspiring, first-person mission for the AI.\n\n"
            "Examples of good creative seeds:\n"
            "- 'Explore the history of human philosophy, from the ancient Greeks to modern existentialism, to understand how humanity has grappled with questions of meaning and purpose.'\n"
            "- 'Investigate the principles of biological evolution. Learn about natural selection, genetics, and the diversity of life to understand the origins of complexity.'\n\n"
            "Now, generate one new, unique, and inspiring creative seed for the AI's next grand learning mission:"
        )

        try:
            response = self._run_llm_completion_with_retries(
                f"[INST]{prompt}[/INST]", max_tokens=256
            )
            if response:
                clean_response = response.strip().strip('"')
                logger.info(
                    f"[Curriculum Generation]: New creative seed generated: '{clean_response}'"
                )
                return clean_response
        except Exception as e:
            logger.error(
                f"[Curriculum Generation]: Failed to generate new creative seed: {e}"
            )

        return "Investigate the concept of 'logic' and its role in reasoning."
