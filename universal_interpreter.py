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
                json.dump(cache_data, f, indent=2)
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
            "You are a highly intelligent text analysis and relation extraction engine. Your task is to analyze the user's input "
            "and convert it into a structured JSON object. Your primary goal is to extract factual relationships or commands. "
            "Your only output should be a single, valid JSON object with no other text before or after it."
        )
        json_structure_prompt = (
            "The JSON object must have the following fields:\n"
            "- 'intent': Classify the user's primary intent. Possible values are: 'greeting', 'farewell', 'question_about_entity', 'statement_of_fact', 'gratitude', 'positive_affirmation', 'command', 'unknown'.\n"
            "- 'entities': A list of named entities...\n"
            "- 'relation': If the intent is 'statement_of_fact', extract the core factual relationship as an object with three fields: 'subject', 'verb', and 'object'. "
            "IMPORTANT: For definitional sentences like 'X is a Y that does Z', the 'object' should be the simplest possible noun phrase (e.g., 'Y'), not the entire descriptive clause.\n"
            "- 'key_topics': A list of the main subjects or topics...\n"
            "- 'full_text_rephrased': A neutral, one-sentence rephrasing..."
        )
        examples_prompt = (
            "Here are some examples:\n"
            # --- FIX: Restored the 'command' example ---
            "Input: 'show all facts'\n"
            'Output: {"intent": "command", "entities": [], "relation": null, "key_topics": ["show all facts"], "full_text_rephrased": "User has issued a command to show all facts."}\n\n'
            "Input: 'A restaurant is an establishment that prepares and serves food.'\n"
            'Output: {"intent": "statement_of_fact", "entities": [{"name": "restaurant", "type": "LOCATION"}, {"name": "establishment", "type": "MISC"}], "relation": {"subject": "A restaurant", "verb": "is", "object": "an establishment"}, "key_topics": ["restaurant", "establishment", "food"], "full_text_rephrased": "User is defining what a restaurant is."}\n\n'
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

    def synthesize(self, structured_facts: str, original_question: str = None) -> str:
        if structured_facts in self.synthesis_cache and not original_question:
            print("  [Synthesizer Cache]: Hit!")
            return self.synthesis_cache[structured_facts]
        print("  [Synthesizer Cache]: Miss. Running LLM for synthesis.")
        
        system_prompt = (
            "You are a language rephrasing engine. Your task is to rephrase the provided 'Facts' into a single, natural English sentence. "
            "You MUST NOT add, infer, or use any external knowledge. You MUST only use the information given in the 'Facts' string. "
            "Your output must be ONLY the rephrased sentence and nothing else."
        )
        task_prompt = f"Facts: '{structured_facts}'"
        if original_question:
            task_prompt = f"Using ONLY the facts provided, answer the following question.\nQuestion: '{original_question}'\nFacts: '{structured_facts}'"
        full_prompt = (
            f"<s>[INST] {system_prompt}\n\nRephrase the following facts into a single sentence, using the original question for context if provided.\n\n{task_prompt}[/INST]"
        )
        try:
            output = self.llm(
                full_prompt, max_tokens=256, stop=["</s>", "\n"], echo=False, 
                temperature=0.1
            )
            synthesized_text = output["choices"][0]["text"].strip()

            if "(" in synthesized_text:
                synthesized_text = synthesized_text.split("(")[0].strip()
            
            if not original_question:
                self.synthesis_cache[structured_facts] = synthesized_text
                self._save_cache()
            return synthesized_text
        except Exception as e:
            print(f"  [Synthesizer Error]: Could not generate fluent text. Error: {e}")
            return structured_facts