from __future__ import annotations

# universal_interpreter.py
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypedDict, cast

from llama_cpp import Llama

if TYPE_CHECKING:
    from typing import NotRequired

MODELS_FOLDER: Final = Path("models")
DEFAULT_MODEL_PATH: Final = MODELS_FOLDER / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEFAULT_CACHE_PATH: Final = "interpreter_cache.json"

REPHRASING_PROMPT: Final = """
You are a language rephrasing engine. Your task is to convert the given 'Facts' into a single, natural English sentence. You are a fluent parrot. You must follow these rules strictly:
1.  **ONLY use the information given in the 'Facts' string.**
2.  **DO NOT add any extra information, commentary, or meta-analysis.**
3.  **DO NOT apologize or mention your own limitations.**
4.  **Your output must be ONLY the rephrased sentence and nothing else.**"""[1:]

JSON_STRUCTURE_PROMPT: Final = """
The JSON object must have the following fields:
- 'intent': Classify the user's primary intent. Possible values are: 'greeting', 'farewell', 'question_about_entity', 'question_about_concept', 'statement_of_fact', 'statement_of_correction', 'gratitude', 'acknowledgment', 'positive_affirmation', 'command', 'unknown'.
- 'relation': If 'statement_of_fact' or 'statement_of_correction', extract the core relationship. This object has fields: 'subject', 'verb', 'object', and an optional 'properties' object.
If the sentence contains temporal information, extract it into a 'properties' object with an 'effective_date' field in YYYY-MM-DD format.
- 'key_topics': A list of the main subjects or topics...
- 'full_text_rephrased': A neutral, one-sentence rephrasing..."""[1:]


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
]


class PropertyData(TypedDict):
    effective_date: str  # YYYY-MM-DD


class RelationData(TypedDict):
    subject: str
    verb: str
    object: str
    properties: NotRequired[PropertyData]


class Entity(TypedDict):
    name: str
    type: Literal["CONCEPT", "PERSON", "ROLE", "PROPERTY"]


class InterpretData(TypedDict):
    intent: Intent
    entities: list[Entity]
    relation: RelationData | None
    key_topics: list[str]
    full_text_rephrased: str


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
        model_path: str | Path = DEFAULT_MODEL_PATH,
        cache_file: str | Path = DEFAULT_CACHE_PATH,
    ) -> None:
        """Initialize the UniversalInterpreter and load the LLM into memory.

        Args:
            model_path: The file path to the GGUF-formatted LLM model.
            cache_file: The file path for the interpretation and synthesis
                caches.
        """
        print("Initializing Universal Interpreter (loading Mini LLM)...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Interpreter model not found at {model_path}. Please download it.",
            )

        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=0,
            n_ctx=2048,
            n_threads=0,
            n_batch=1024,
            verbose=False,
        )

        self.interpretation_cache: dict[str, InterpretData] = {}
        self.synthesis_cache: dict[str, str] = {}

        self.cache_file = cache_file
        self._load_cache()

        print("Universal Interpreter loaded successfully.")

    def _load_cache(self) -> None:
        """Load the interpretation and synthesis caches from a JSON file."""
        if not os.path.exists(self.cache_file):
            print("[Cache]: No cache file found. Starting fresh.")
            return

        try:
            with open(self.cache_file) as f:
                cache_data = json.load(f)
                self.interpretation_cache = dict(
                    cache_data.get("interpretations", []),
                )
                self.synthesis_cache = dict(cache_data.get("synthesis", []))
            print(
                f"[Cache]: Loaded {len(self.interpretation_cache)} interpretation(s) and {len(self.synthesis_cache)} synthesis caches from {self.cache_file}.",
            )
        except Exception as e:
            print(
                f"[Cache Error]: Could not load cache file. Starting fresh. Error: {e}",
            )

    def _save_cache(self) -> None:
        """Save the current interpretation and synthesis caches to a JSON file."""
        try:
            with open(self.cache_file, "w") as f:
                cache_data = {
                    "interpretations": list(self.interpretation_cache.items()),
                    "synthesis": list(self.synthesis_cache.items()),
                }
                json.dump(cache_data, f, indent=4)
        except Exception as e:
            print(
                f"[Cache Error]: Could not save cache to {self.cache_file}. Error: {e}",
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
        return re.sub(r'"\s*\n\s*"', '", "', json_str)

    def interpret(self, user_input: str) -> InterpretData:
        """Analyze user input with the LLM and return a structured interpretation.

        This is the core, context-free interpretation method. It constructs
        a detailed prompt with examples and instructions, sends it to the
        LLM, and parses the resulting JSON output into a `InterpretData`
        TypedDict. Results are cached for performance.

        Args:
            user_input: The raw user message to be interpreted.

        Returns:
            A `InterpretData` object representing the structured
            understanding of the input. Returns a default 'unknown'
            intent on failure.
        """
        cache_key = user_input
        if cache_key in self.interpretation_cache:
            print("  [Interpreter Cache]: Hit!")
            return self.interpretation_cache[cache_key]
        print("  [Interpreter Cache]: Miss. Running LLM for interpretation.")

        system_prompt = (
            "You are a strict, precise text analysis engine. Your only task is to analyze user input "
            "and convert it into a structured JSON object. Extract factual relationships or commands. "
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
            f"<s>[INST] {system_prompt}\n\n{json_structure_prompt}\n\n{examples_prompt}\n\n"
            f"Now, analyze the following user input and provide ONLY the JSON output:\n{sanitized_input}[/INST]"
        )
        try:
            output = cast(
                "dict",
                self.llm(
                    full_prompt,
                    max_tokens=512,
                    stop=["</s>"],
                    echo=False,
                    temperature=0.0,
                ),
            )
            assert isinstance(output, dict)
            response_text = output["choices"][0]["text"].strip()
            cleaned_json_str = self._clean_llm_json_output(response_text)
            if not cleaned_json_str:
                raise json.JSONDecodeError("No JSON object found", response_text, 0)
            raw_interpretation = json.loads(cleaned_json_str)

            interpretation = cast("InterpretData", raw_interpretation)

            self.interpretation_cache[cache_key] = interpretation
            self._save_cache()
            return interpretation
        except Exception as e:
            print(f"  [Interpreter Error]: Could not parse LLM output. Error: {e}")
            return InterpretData(
                {
                    "intent": "unknown",
                    "entities": [],
                    "relation": None,
                    "key_topics": user_input.split(),
                    "full_text_rephrased": f"Could not fully interpret: '{user_input}'",
                },
            )

    def resolve_context(self, history: list[str], new_input: str) -> str:
        """Use the LLM to perform coreference resolution on the user's input.

        This method attempts to replace pronouns in the user's latest
        message (e.g., "it", "they") with the specific nouns they refer
        to from the preceding conversation history. This creates a
        context-aware input for the main `interpret` method.

        Args:
            history: A list of the previous turns in the conversation.
            new_input: The user's latest message, potentially containing
                pronouns.

        Returns:
            The rephrased input string with pronouns resolved, or the
            original input if no changes were needed.
        """
        print("  [Context Resolver]: Attempting to resolve pronouns...")
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
            f"<s>[INST] {system_prompt}\n\n{examples_prompt}\n\n"
            f"Conversation History:\n{formatted_history}\n"
            f"New Input: {new_input}\nOutput:[/INST]"
        )
        try:
            output = cast(
                "dict",
                self.llm(
                    full_prompt,
                    max_tokens=128,
                    stop=["</s>", "\n"],
                    echo=False,
                    temperature=0.0,
                ),
            )
            assert isinstance(output, dict)
            rephrased_input = output["choices"][0]["text"].strip()
            if rephrased_input and rephrased_input.lower() != new_input.lower():
                print(f"    - Context resolved: '{new_input}' -> '{rephrased_input}'")
                return cast("str", rephrased_input)
            print("    - No context to resolve, using original input.")
            return new_input
        except Exception as e:
            print(f"  [Context Resolver Error]: Could not resolve context. Error: {e}")
            return new_input

    def interpret_with_context(
        self,
        user_input: str,
        history: list[str],
    ) -> InterpretData:
        """Interpret user input after first attempting to resolve context.

        This is a wrapper method that orchestrates contextual interpretation.
        It first checks if the input contains pronouns and, if so, calls
        the `resolve_context` method before passing the potentially
        rephrased input to the main `interpret` method.

        Args:
            user_input: The raw message from the user.
            history: The preceding conversation history.

        Returns:
            A `InterpretData` object representing the structured
            understanding of the contextualized input.
        """
        contextual_input = user_input
        pronouns = ["it", "its", "they", "them", "their", "he", "she", "his", "her"]
        if history and any(
            f" {pronoun} " in f" {user_input.lower()} " for pronoun in pronouns
        ):
            contextual_input = self.resolve_context(history, user_input)
        return self.interpret(contextual_input)

    def generate_curious_questions(self, topic: str, known_fact: str) -> list[str]:
        """Generate simple, fundamental follow-up questions about a topic.

        This method uses the LLM in a more creative mode to simulate
        curiosity. Given a known fact, it generates two simple questions
        that a child might ask to learn more, which can then be used by
        the `KnowledgeHarvester` to guide its study process.

        Args:
            topic: The high-level topic being studied.
            known_fact: A single, declarative fact that is already known.

        Returns:
            A list of two generated question strings, or an empty list on failure.
        """
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
            f"<s>[INST] {system_prompt}\n\n{examples_prompt}\n\n"
            f"Topic: {topic}\nKnown Fact: {known_fact}\n"
            f"Output:[/INST]"
        )
        try:
            output = cast(
                "dict",
                self.llm(
                    full_prompt,
                    max_tokens=128,
                    stop=["</s>"],
                    echo=False,
                    temperature=0.8,
                ),
            )
            assert isinstance(output, dict)
            response_text = output["choices"][0]["text"].strip()
            questions = [
                q.strip()
                for q in response_text.replace("-", "").split("\n")
                if q.strip()
            ]

            if questions:
                print(f"    - Generated {len(questions)} curious questions.")
                return questions
            return []
        except Exception as e:
            print(
                f"  [Question Generation Error]: Could not generate questions. Error: {e}",
            )
            return []

    def synthesize(
        self,
        structured_facts: str,
        original_question: str | None = None,
        mode: str = "statement",
    ) -> str:
        """Convert a structured, internal representation into natural language.

        This is the "voice" of the agent. It takes a structured string of
        facts or an internal state and uses the LLM to generate a fluent,
        conversational sentence. It can operate in different modes, such
        as generating a statement or a clarification question. Results
        are cached for performance.

        Args:
            structured_facts: The internal data to be verbalized.
            original_question: The user's question, used for context.
            mode: The synthesis mode ('statement' or 'clarification_question').

        Returns:
            A natural language string representing the synthesized response.
        """
        cache_key = f"{mode}|{original_question}|{structured_facts}"

        if cache_key in self.synthesis_cache:
            print("  [Synthesizer Cache]: Hit!")
            return self.synthesis_cache[cache_key]

        print(
            f"  [Synthesizer Cache]: Miss. Running LLM for synthesis in '{mode}' mode.",
        )

        system_prompt = ""
        task_prompt = ""

        if mode == "clarification_question":
            system_prompt = (
                "You are an inquisitive AI agent. Your task is to ask a clarifying question. "
                "You have been given two conflicting facts. Formulate a single, polite, and simple question. "
                "Do not state the facts directly. Your output must be ONLY the question."
            )
            task_prompt = f"Conflicting Facts: '{structured_facts}'"
        else:
            system_prompt = REPHRASING_PROMPT
            task_prompt = f"Facts to rephrase: '{structured_facts}'"
            if original_question:
                task_prompt = f"Using ONLY the facts provided, directly answer the question.\nQuestion: '{original_question}'\nFacts: '{structured_facts}'"

        full_prompt = f"<s>[INST] {system_prompt}\n\n{task_prompt}[/INST]"
        try:
            output = cast(
                "dict",
                self.llm(
                    full_prompt,
                    max_tokens=256,
                    stop=["</s>", "\n"],
                    echo=False,
                    temperature=0.7 if mode == "clarification_question" else 0.1,
                ),
            )
            assert isinstance(output, dict)
            synthesized_text = output["choices"][0]["text"].strip().replace('"', "")
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

            return cast("str", synthesized_text)
        except Exception as e:
            print(f"  [Synthesizer Error]: Could not generate fluent text. Error: {e}")
            return structured_facts
