# universal_interpreter.py

import json
import os
import re
from llama_cpp import Llama

class UniversalInterpreter:
    def __init__(self, model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        print("Initializing Universal Interpreter (loading Mini LLM)...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Interpreter model not found at {model_path}. Please download it.")
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=0, n_ctx=2048, n_threads=0, n_batch=1024, verbose=False
        )
        
        self.cache_file = "interpreter_cache.json"
        self._load_cache()
        
        print("Universal Interpreter loaded successfully.")
        
    def _load_cache(self):
        """Loads the interpretation and synthesis caches from a JSON file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.interpretation_cache = dict(cache_data.get("interpretations", []))
                    self.synthesis_cache = dict(cache_data.get("synthesis", []))
                print(f"[Cache]: Loaded {len(self.interpretation_cache)} interpretation(s) and {len(self.synthesis_cache)} synthesis caches from {self.cache_file}.")
            except Exception as e:
                print(f"[Cache Error]: Could not load cache file. Starting fresh. Error: {e}")
                self.interpretation_cache, self.synthesis_cache = {}, {}
        else:
            print("[Cache]: No cache file found. Starting fresh.")
            self.interpretation_cache, self.synthesis_cache = {}, {}

    def _save_cache(self):
        """Saves the current caches to a JSON file."""
        try:
            with open(self.cache_file, 'w') as f:
                cache_data = {
                    "interpretations": list(self.interpretation_cache.items()),
                    "synthesis": list(self.synthesis_cache.items())
                }
                json.dump(cache_data, f, indent=4)
        except Exception as e:
            print(f"[Cache Error]: Could not save cache to {self.cache_file}. Error: {e}")
            
    def _clean_llm_json_output(self, raw_text: str) -> str:
        """More robustly cleans the LLM's output to extract a valid JSON object."""
        start_brace = raw_text.find('{')
        end_brace = raw_text.rfind('}')
        if start_brace == -1 or end_brace == -1: return ""
        json_str = raw_text[start_brace:end_brace+1]
        json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
        json_str = re.sub(r'"\s*\n\s*"', '", "', json_str)
        return json_str

    def interpret(self, user_input: str) -> dict:
        """
        Uses the Mini LLM to analyze user input and return a structured JSON.
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
        # --- UPDATED: Added 'properties' to the relation object description ---
        json_structure_prompt = (
            "The JSON object must have the following fields:\n"
            "- 'intent': Classify the user's primary intent...\n"
            "- 'entities': A list of named entities...\n"
            "- 'relation': If 'statement_of_fact', extract the core relationship. This object has fields: 'subject', 'verb', 'object', and an optional 'properties' object. "
            "If the sentence contains temporal information (like a date or year), extract it into a 'properties' object with an 'effective_date' field in YYYY-MM-DD format.\n"
            "- 'key_topics': A list of the main subjects or topics...\n"
            "- 'full_text_rephrased': A neutral, one-sentence rephrasing..."
        )
        # --- UPDATED: Added a new example for temporal facts ---
        examples_prompt = (
            "Here are some examples:\n"
            "Input: 'show all facts'\n"
            'Output: {"intent": "command", "entities": [], "relation": null, "key_topics": ["show all facts"], "full_text_rephrased": "User has issued a command to show all facts."}\n\n'
            "Input: 'In 2023, Tim Cook was the CEO of Apple.'\n"
            'Output: {"intent": "statement_of_fact", "entities": [{"name": "Tim Cook", "type": "PERSON"}, {"name": "CEO of Apple", "type": "ROLE"}], "relation": {"subject": "Tim Cook", "verb": "was", "object": "the CEO of Apple", "properties": {"effective_date": "2023-01-01"}}, "key_topics": ["Tim Cook", "Apple", "CEO"], "full_text_rephrased": "User is stating that Tim Cook was the CEO of Apple in 2023."}\n\n'
            "Input: 'a fact is a piece of information'\n"
            'Output: {"intent": "statement_of_fact", "entities": [{"name": "fact", "type": "CONCEPT"}, {"name": "piece of information", "type": "CONCEPT"}], "relation": {"subject": "a fact", "verb": "is", "object": "a piece of information"}, "key_topics": ["fact", "information"], "full_text_rephrased": "User is stating that a fact is a piece of information."}\n\n'
            "Input: 'who is Donald Trump?'\n"
            'Output: {"intent": "question_about_entity", "entities": [{"name": "Donald Trump", "type": "PERSON"}], "relation": null, "key_topics": ["Donald Trump"], "full_text_rephrased": "User is asking for information about Donald Trump."}\n'
        )
        
        full_prompt = (
            f"<s>[INST] {system_prompt}\n\n{json_structure_prompt}\n\n{examples_prompt}\n\n"
            f"Now, analyze the following user input and provide ONLY the JSON output:\n'{user_input}'[/INST]"
        )
        try:
            output = self.llm(full_prompt, max_tokens=512, stop=["</s>"], echo=False, temperature=0.0)
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
            return {"intent": "unknown", "entities": [], "relation": None, "key_topics": user_input.split(), "full_text_rephrased": f"Could not fully interpret: '{user_input}'"}

    def synthesize(self, structured_facts: str, original_question: str = None, mode: str = "statement") -> str:
        """
        Uses the Mini LLM to convert structured facts into natural language.
        Can operate in different modes: 'statement' (default) or 'clarification_question'.
        """
        if mode == "statement" and structured_facts in self.synthesis_cache and not original_question:
            print("  [Synthesizer Cache]: Hit!")
            return self.synthesis_cache[structured_facts]
        print(f"  [Synthesizer Cache]: Miss. Running LLM for synthesis in '{mode}' mode.")
        
        system_prompt = ""
        task_prompt = ""
        
        if mode == "clarification_question":
            system_prompt = (
                "You are an inquisitive AI agent. Your task is to ask a clarifying question to a human user. "
                "You have been given two conflicting facts. Formulate a single, polite, and simple question that will help you "
                "determine the correct information. Do not state the facts directly. Your output must be ONLY the question."
            )
            task_prompt = f"Conflicting Facts: '{structured_facts}'"
        else: # Default "statement" mode
            system_prompt = (
                "You are a STICKT language rephrasing engine. Your task is to convert the given 'Facts' into a single, grammatically correct, natural English sentence. "
                "You are a fluent parrot. You MUST NOT add any extra information, commentary, or meta-analysis. "
                "You MUST NOT apologize or mention limitations. "
                "You MUST ONLY use the information given in the 'Facts' string. "
                "Your output must be ONLY the rephrased sentence and nothing else."
            )
            task_prompt = f"Facts to rephrase: '{structured_facts}'"
            if original_question:
                task_prompt = f"Using ONLY the facts provided, directly answer the question.\nQuestion: '{original_question}'\nFacts: '{structured_facts}'"
        
        full_prompt = (
            f"<s>[INST] {system_prompt}\n\n{task_prompt}[/INST]"
        )
        try:
            output = self.llm(
                full_prompt, max_tokens=256, stop=["</s>", "\n"], echo=False, 
                temperature=0.7 if mode == "clarification_question" else 0.1
            )
            synthesized_text = output["choices"][0]["text"].strip().replace('"', '')

            phrases_to_remove = ["rephrased sentence:", "based on the provided facts,", "the rephrased sentence is:"]
            for phrase in phrases_to_remove:
                if synthesized_text.lower().startswith(phrase):
                    synthesized_text = synthesized_text[len(phrase):].strip()

            if "(" in synthesized_text:
                synthesized_text = synthesized_text.split("(")[0].strip()
            
            if mode == "statement" and not original_question:
                self.synthesis_cache[structured_facts] = synthesized_text
                self._save_cache()
            return synthesized_text
        except Exception as e:
            print(f"  [Synthesizer Error]: Could not generate fluent text. Error: {e}")
            return structured_facts