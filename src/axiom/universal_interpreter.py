from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import (
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    from .graph_core import ConceptNode

logger = logging.getLogger(__name__)

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
        """Initialize the UniversalInterpreter and load the LLM into memory.

        Args:
            model_path: The file path to the GGUF-formatted LLM model.
            cache_file: The file path for the interpretation and synthesis
                caches.
        """
        logger.info(
            "[yellow]Initializing Universal Interpreter (loading Mini LLM if enabled)...[/yellow]"
        )
        self.llm: Llama | None = None

        if load_llm:
            logger.info(
                "[yellow]Initializing Universal Interpreter (loading Mini LLM)...[/yellow]"
            )
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"[error]Interpreter model not found at {model_path}.[/error] [yellow]Please download using axiom-llm .[/yelow]",
                )

            with open(os.devnull, "w") as f, redirect_stderr(f):
                self.llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=0,
                    n_ctx=16384,
                    n_threads=0,
                    n_batch=1024,
                    verbose=False,
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

    def _clean_llm_json_output(self, raw_text: str) -> str:
        """Clean and extract a JSON object from the raw output of an LLM.

        This utility handles common LLM output issues, such as extraneous
        text before or after the JSON object and trailing commas that
        would cause parsing errors.

        Args:
            raw_text: The raw string response from the LLM.

        Returns:
            A string containing only the cleaned JSON object, or an empty
            string if no JSON object could be found.
        """
        start_brace = raw_text.find("{")
        end_brace = raw_text.rfind("}")
        if start_brace == -1 or end_brace == -1:
            return ""
        json_str = raw_text[start_brace : end_brace + 1]
        json_str = re.sub(r",\s*(\}|\])", r"\1", json_str)
        return json_str.replace("\n", ", ")

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

    def decompose_sentence_to_relations(
        self, text: str, main_topic: str | None = None
    ) -> list[RelationData]:
        """
        Uses the LLM to decompose a sentence into a list of atomic relations.
        This is the primary method for learning from complex sentences.
        """
        if self.llm is None:
            logger.warning("[Interpreter]: LLM is disabled. Cannot decompose text.")
            return []

        logger.info(
            "[purple]   - [Interpreter]: Decomposing sentence into atomic facts...[/purple]"
        )

        topic_context = (
            f"The primary topic of this sentence is '{main_topic}'. "
            "Ensure the subject of at least one core relation is this topic or a direct synonym."
            if main_topic
            else ""
        )

        prompt = f"""
        **ROLE:** You are a knowledge engineering system. Your task is to extract and decompose knowledge from a sentence into a list of simple, atomic semantic relations.

        **TASK:** Analyze the user's sentence and break it down into multiple, simple, atomic relations.
        - Each relation MUST be a JSON object: {{"subject": "...", "verb": "...", "object": "..."}}
        - Subjects and objects should be simple concepts (e.g., "cats", "photosynthesis", "the sun").
        - Verbs should be concise, standardized predicates (e.g., "is_a", "has_property", "causes", "is_part_of"). Use snake_case.
        - The goal is DECOMPOSITION. One complex sentence should become several simple facts.

        **CONTEXT:** {topic_context}

        **EXAMPLE:**
        Topic: Photosynthesis
        Sentence: "Photosynthesis is a process used by plants to convert light energy into chemical energy."
        Output:
        [
          {{"subject": "photosynthesis", "verb": "is_a", "object": "process"}},
          {{"subject": "photosynthesis", "verb": "is_used_by", "object": "plants"}},
          {{"subject": "photosynthesis", "verb": "converts", "object": "light energy"}},
          {{"subject": "light energy", "verb": "is_converted_into", "object": "chemical energy"}}
        ]

        **SENTENCE TO ANALYZE:**
        "{text}"

        **RULES:**
        1.  Return ONLY a valid JSON list of relation objects.
        2.  Do NOT include markdown, explanations, or any other text outside the JSON list.
        3.  If the sentence contains no extractable facts, return an empty list `[]`.
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
                    "  [Interpreter Warning]: LLM response did not contain a JSON list. Output: %s",
                    response_text,
                )
                return []

            json_str = response_text[start_bracket : end_bracket + 1]
            try:
                relations = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(
                    "  [Interpreter Error]: Failed to decompose sentence. Error: %s", e
                )
                logger.debug(
                    "  [Interpreter Debug]: Malformed JSON from LLM:\n%s", json_str
                )
                return []

            if isinstance(relations, list):
                logger.info(
                    "[green]   - Decomposed sentence into %d atomic relations.[/green]",
                    len(relations),
                )
                return cast("list[RelationData]", relations)

            logger.warning(
                "  [Interpreter Warning]: LLM returned valid JSON, but it was not a list. Output: %s",
                json_str,
            )
            return []

        except Exception as e:
            logger.error(
                "  [Interpreter Error]: An unexpected error occurred during decomposition. Error: %s",
                e,
            )
            return []

    def break_down_definition(self, subject: str, chunky_definition: str) -> list[str]:
        """Use the LLM to break a complex definition into simple, atomic facts."""
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
                "  [Interpreter Error]: Could not break down definition. Error: %s", e
            )
            return []

    def generate_curious_questions(self, topic: str, known_fact: str) -> list[str]:
        """Generate simple, fundamental follow-up questions about a topic."""
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
        if self.llm is None:
            logger.warning(
                "  [Fact Verifier Error]: LLM is disabled. Cannot verify/reframe."
            )
            if original_topic.lower() in raw_sentence.lower():
                return raw_sentence
            return None

        logger.info(
            "[purple]   Asking LLM to verify and reframe fact for '%s'...[/purple]",
            original_topic,
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
                logger.error("  [Fact Verifier Error]: LLM returned an empty response.")
                return None

            if "none" in rephrased_fact.lower():
                logger.info("    - LLM rejected the fact as irrelevant.")
                return None

            logger.info(
                "[success]   - LLM verified and reframed: '%s'[/success]",
                rephrased_fact,
            )
            return rephrased_fact

        except Exception as e:
            logger.error(
                "  [Fact Verifier Error]: Could not process fact with LLM. Error: %s",
                e,
            )
            return None

    def synthesize(
        self,
        structured_facts: str | list[RelationData] | list[str] | list[ConceptNode],
        original_question: str | None = None,
        mode: str = "statement",
    ) -> str:
        """Convert a structured, internal representation into natural language."""
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
            temperature = 0.7 if mode == "clarification_question" else 0.1
            synthesized_text = self._run_llm_completion_with_retries(
                full_prompt, max_tokens=256, temperature=temperature
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
                "  [Synthesizer Error]: Could not generate fluent text. Error: %s", e
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
