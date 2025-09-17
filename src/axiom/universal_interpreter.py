# universal_interpreter.py

import json
import os
import re

from llama_cpp import Llama


class UniversalInterpreter:
    def __init__(self, model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        print("Initializing Universal Interpreter (loading Mini LLM)...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Interpreter model not found at {model_path}. Please download it.",
            )

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=0,
            n_ctx=2048,
            n_threads=0,
            n_batch=1024,
            verbose=False,
        )

        self.cache_file = "interpreter_cache.json"
        self._load_cache()

        print("Universal Interpreter loaded successfully.")

    def _load_cache(self):
        """Loads the interpretation and synthesis caches from a JSON file."""
        if os.path.exists(self.cache_file):
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
                self.interpretation_cache, self.synthesis_cache = {}, {}
        else:
            print("[Cache]: No cache file found. Starting fresh.")
            self.interpretation_cache, self.synthesis_cache = {}, {}

    def _save_cache(self):
        """Saves the current caches to a JSON file."""
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
        """More robustly cleans the LLM's output to extract a valid JSON object."""
        start_brace = raw_text.find("{")
        end_brace = raw_text.rfind("}")
        if start_brace == -1 or end_brace == -1:
            return ""
        json_str = raw_text[start_brace : end_brace + 1]
        json_str = re.sub(r",\s*(\}|\])", r"\1", json_str)
        return re.sub(r'"\s*\n\s*"', '", "', json_str)

    def interpret(self, user_input: str) -> dict:
        """
        Uses the Mini LLM to analyze user input and return a structured JSON.
        This is the core, context-free interpretation method.
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
        json_structure_prompt = (
            "The JSON object must have the following fields:\n"
            "- 'intent': Classify the user's primary intent. Possible values are: 'greeting', 'farewell', 'question_about_entity', 'question_about_concept', 'statement_of_fact', 'statement_of_correction', 'gratitude', 'acknowledgment', 'positive_affirmation', 'command', 'unknown'.\n"
            "- 'relation': If 'statement_of_fact' or 'statement_of_correction', extract the core relationship. This object has fields: 'subject', 'verb', 'object', and an optional 'properties' object. "
            "If the sentence contains temporal information, extract it into a 'properties' object with an 'effective_date' field in YYYY-MM-DD format.\n"
            "- 'key_topics': A list of the main subjects or topics...\n"
            "- 'full_text_rephrased': A neutral, one-sentence rephrasing..."
        )

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
            output = self.llm(
                full_prompt,
                max_tokens=512,
                stop=["</s>"],
                echo=False,
                temperature=0.0,
            )
            response_text = output["choices"][0]["text"].strip()
            cleaned_json_str = self._clean_llm_json_output(response_text)
            if not cleaned_json_str:
                raise json.JSONDecodeError("No JSON object found", response_text, 0)
            interpretation = json.loads(cleaned_json_str)
            self.interpretation_cache[cache_key] = interpretation
            self._save_cache()
            return interpretation
        except Exception as e:
            print(f"  [Interpreter Error]: Could not parse LLM output. Error: {e}")
            return {
                "intent": "unknown",
                "entities": [],
                "relation": None,
                "key_topics": user_input.split(),
                "full_text_rephrased": f"Could not fully interpret: '{user_input}'",
            }

    def resolve_context(self, history: list[str], new_input: str) -> str:
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
            output = self.llm(
                full_prompt,
                max_tokens=128,
                stop=["</s>", "\n"],
                echo=False,
                temperature=0.0,
            )
            rephrased_input = output["choices"][0]["text"].strip()
            if rephrased_input and rephrased_input.lower() != new_input.lower():
                print(f"    - Context resolved: '{new_input}' -> '{rephrased_input}'")
                return rephrased_input
            print("    - No context to resolve, using original input.")
            return new_input
        except Exception as e:
            print(f"  [Context Resolver Error]: Could not resolve context. Error: {e}")
            return new_input

    def interpret_with_context(self, user_input: str, history: list[str]) -> dict:
        contextual_input = user_input
        pronouns = ["it", "its", "they", "them", "their", "he", "she", "his", "her"]
        if history and any(
            f" {pronoun} " in f" {user_input.lower()} " for pronoun in pronouns
        ):
            contextual_input = self.resolve_context(history, user_input)
        return self.interpret(contextual_input)

    def generate_curious_questions(self, topic: str, known_fact: str) -> list[str]:
        """
        Uses the Mini LLM in a creative mode to generate follow-up questions.
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
            output = self.llm(
                full_prompt,
                max_tokens=128,
                stop=["</s>"],
                echo=False,
                temperature=0.8,
            )
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

    # --- METHOD UPDATED WITH FULL CACHING LOGIC ---
    def synthesize(
        self,
        structured_facts: str,
        original_question: str = None,
        mode: str = "statement",
    ) -> str:
        """
        Uses the Mini LLM to convert structured facts into natural language.
        Results are cached to avoid repeated LLM calls for the same synthesis task.
        """
        # --- CACHING FIX: Use a more robust cache key ---
        # The original question makes the context unique, so it must be part of the key.
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
        else:  # Default "statement" mode
            system_prompt = (
                "You are a language rephrasing engine. Your task is to convert the given 'Facts' into a single, natural English sentence. "
                "You are a fluent parrot. You must follow these rules strictly:\n"
                "1.  **ONLY use the information given in the 'Facts' string.**\n"
                "2.  **DO NOT add any extra information, commentary, or meta-analysis.**\n"
                "3.  **DO NOT apologize or mention your own limitations.**\n"
                "4.  **Your output must be ONLY the rephrased sentence and nothing else.**"
            )
            task_prompt = f"Facts to rephrase: '{structured_facts}'"
            if original_question:
                task_prompt = f"Using ONLY the facts provided, directly answer the question.\nQuestion: '{original_question}'\nFacts: '{structured_facts}'"

        full_prompt = f"<s>[INST] {system_prompt}\n\n{task_prompt}[/INST]"
        try:
            output = self.llm(
                full_prompt,
                max_tokens=256,
                stop=["</s>", "\n"],
                echo=False,
                temperature=0.7 if mode == "clarification_question" else 0.1,
            )
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

            # --- CACHING FIX: Save the result to the cache before returning ---
            self.synthesis_cache[cache_key] = synthesized_text
            self._save_cache()

            return synthesized_text
        except Exception as e:
            print(f"  [Synthesizer Error]: Could not generate fluent text. Error: {e}")
            return structured_facts
