# knowledge_harvester.py

import time
import random
import requests
import wikipedia
import os
import json
from datetime import datetime # Import the datetime module

class KnowledgeHarvester:
    def __init__(self, agent, lock):
        self.agent = agent
        self.lock = lock
        self.nyt_api_key = os.environ.get("NYT_API_KEY")
        if not self.nyt_api_key:
            print("[Knowledge Harvester WARNING]: NYT_API_KEY environment variable not set. Trending topic source will be disabled.")
        
        self.rejected_topics = set()
        self.enable_anticipatory_cache = False
        
        wikipedia.set_user_agent('AxiomAgent/1.0 (AxiomAgent@example.com)')
        print("[Knowledge Harvester]: Initialized.")

    def _is_sentence_simple_enough(self, sentence: str, max_words=40, max_commas=3) -> bool:
        """Checks if a sentence is simple enough for the interpreter to handle reliably."""
        if len(sentence.split()) > max_words:
            print(f"  [Simplicity Filter]: Sentence rejected (too long: {len(sentence.split())} words).")
            return False
        if sentence.count(',') > max_commas:
            print(f"  [Simplicity Filter]: Sentence rejected (too complex: {sentence.count(',')} commas).")
            return False
        return True

    def _extract_core_entity(self, topic_string: str) -> str | None:
        print(f"  [Harvester]: Analyzing topic string for core entity: '{topic_string}'")
        try:
            interpretation = self.agent.interpreter.interpret(topic_string)
            entities = interpretation.get('entities', [])
            if entities:
                priority_types = ['PERSON', 'ORGANIZATION', 'LOCATION', 'GPE', 'PRODUCT', 'EVENT']
                for ent_type in priority_types:
                    for entity in entities:
                        if entity.get('type') == ent_type:
                            core_entity = entity.get('name')
                            print(f"  [Harvester]: Extracted core entity: '{core_entity}'")
                            return core_entity
                core_entity = entities[0].get('name')
                print(f"  [Harvester]: Extracted core entity (fallback): '{core_entity}'")
                return core_entity
        except Exception as e:
            print(f"  [Harvester]: Error during core entity extraction: {e}")
        return None

    def find_new_topic(self, max_attempts=10) -> str | None:
        for i in range(max_attempts):
            print(f"\n[Harvester]: Searching for a new topic (Attempt {i + 1}/{max_attempts})...")
            topic = None
            source_choice = random.choice(['nyt', 'wiki', 'wiki'])
            
            # --- UPDATED: Call the new archival method ---
            if source_choice == 'nyt' and self.nyt_api_key:
                topic = self.get_archival_topic()
            else:
                topic = self.get_random_wikipedia_topic()
            
            if topic:
                clean_topic = self.agent._clean_phrase(topic)
                if self.agent.graph.get_node_by_name(clean_topic):
                    print(f"  [Harvester Info]: Agent already knows about '{topic}'. Finding a new topic...")
                    continue
                elif clean_topic in self.rejected_topics:
                    print(f"  [Harvester Info]: Agent has already tried and failed to learn about '{topic}'. Finding a new topic...")
                    continue
                else:
                    print(f"  [Harvester Success]: Found a new, unknown topic: '{topic}'")
                    return topic
        
        print(f"[Harvester Warning]: Could not find a new, unknown topic after {max_attempts} attempts.")
        return None

    # --- DELETED: The old get_trending_topic method is gone ---

    # --- NEW: The archival topic discovery method ---
    def get_archival_topic(self) -> str | None:
        if not self.nyt_api_key: return None
        print("  [Harvester]: Attempting to fetch a random topic from New York Times Archives...")
        try:
            # Pick a random year and month
            current_year = datetime.now().year
            year = random.randint(2000, current_year)
            month = random.randint(1, 12)
            
            print(f"    - Searching archive for {year}-{month:02d}...")
            
            url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={self.nyt_api_key}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            docs = data.get('response', {}).get('docs', [])
            if not docs:
                print("    - No articles found for this month.")
                return None
            
            # Find a random, suitable headline
            random.shuffle(docs)
            for article in docs:
                headline = article.get('headline', {}).get('main', '')
                if headline and len(headline.split()) > 3: # Ensure headline is reasonably long
                    return self._extract_core_entity(headline)

        except requests.RequestException as e:
            print(f"  [Harvester]: Error fetching NYT Archive API: {e}")
        return None

    def get_random_wikipedia_topic(self) -> str | None:
        import wikipediaapi
        wiki_api = wikipediaapi.Wikipedia('AxiomAgent/1.0 (AxiomAgent@example.com)', 'en')
        print("  [Harvester]: Attempting to find a random topic from Wikipedia categories...")
        major_categories = ["History", "Science", "Technology", "Art", "Geography", "Mathematics", "Philosophy", "Culture", "Health", "Nature", "People", "Society"]
        try:
            category_name = random.choice(major_categories)
            category_page = wiki_api.page(f"Category:{category_name}")
            if category_page.exists() and hasattr(category_page, 'categorymembers'):
                members = list(category_page.categorymembers.values())
                pages = [m for m in members if m.ns == wikipediaapi.Namespace.MAIN]
                if pages:
                    chosen_page = random.choice(pages)
                    if not any(keyword in chosen_page.title.lower() for keyword in ['list of', 'index of', 'timeline of', 'outline of']):
                        return chosen_page.title
        except Exception as e:
            print(f"  [Harvester]: Error during Wikipedia search: {e}")
        return None

    def get_fact_from_wikipedia(self, topic: str) -> tuple[str, str] | None:
        print(f"[Knowledge Harvester]: Performing intelligent search for '{topic}' on Wikipedia...")
        try:
            search_results = wikipedia.search(topic, results=5)
            if not search_results:
                print(f"  [Wiki Search]: No results found for '{topic}'.")
                return None
            
            page_title = search_results[0]
            print(f"  [Wiki Search]: Found top result: '{page_title}'. Fetching page...")
            page = wikipedia.page(page_title, auto_suggest=False, redirect=True)
            
            if page and page.summary:
                first_sentence = page.summary.split('. ')[0].strip()
                if not first_sentence.endswith('.'): first_sentence += '.'
                
                if len(first_sentence.split()) > 5 and self._is_sentence_simple_enough(first_sentence):
                    final_topic = page.title
                    print(f"[Knowledge Harvester]: Extracted fact from Wikipedia: '{first_sentence}'")
                    return final_topic, first_sentence
                else:
                    return None
        
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  [Wiki Search]: Hit a disambiguation page for '{topic}'. Discarding.")
        except wikipedia.exceptions.PageError:
            print(f"  [Wiki Search]: PageError. Could not find a specific page for '{topic}'.")
        except Exception as e:
            print(f"  [Wiki Search]: An unexpected error occurred: {e}")

        print(f"[Knowledge Harvester]: Could not find a suitable fact page for '{topic}' on Wikipedia.")
        return None

    def get_fact_from_duckduckgo(self, topic: str) -> tuple[str, str] | None:
        search_query = f"what is {topic}"
        print(f"[Knowledge Harvester]: Fact was unsuitable or failed. Falling back to DuckDuckGo with query: '{search_query}'...")
        try:
            url = f"https://api.duckduckgo.com/?q={search_query}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            definition = data.get('AbstractText') or data.get('Definition')
            if definition:
                fact_sentence = definition.split('. ')[0].strip()
                if not fact_sentence.endswith('.'): fact_sentence += '.'
                
                if not topic.lower() in fact_sentence.lower():
                    fact_sentence = f"{topic.capitalize()} is {fact_sentence.lower()}"

                if "is a redirect to" in fact_sentence.lower(): return None
                
                if self._is_sentence_simple_enough(fact_sentence):
                    print(f"[Knowledge Harvester]: Extracted fact from DuckDuckGo: '{fact_sentence}'")
                    return topic, fact_sentence
                else:
                    return None
        except requests.RequestException as e:
            print(f"[Knowledge Harvester]: Error fetching from DuckDuckGo API: {e}")
        except json.JSONDecodeError:
            print("[Knowledge Harvester]: Error decoding JSON from DuckDuckGo API.")
        return None

    def _try_to_learn(self, fact_sentence: str) -> bool:
        if not fact_sentence: return False
        with self.lock:
            print("  [Lock]: Harvester acquired lock.")
            learned_successfully = self.agent.learn_new_fact_autonomously(fact_sentence)
        print("  [Lock]: Harvester released lock.")
        return learned_successfully

    def harvest_and_learn(self):
        print("\n--- [Autonomous Learning Cycle Started] ---")
        
        max_cycles = 3
        learned_this_interval = False

        for cycle_attempt in range(max_cycles):
            if learned_this_interval: break
            
            print(f"--- Main learning cycle attempt {cycle_attempt + 1}/{max_cycles} ---")
            initial_topic = None
            
            with self.lock:
                print("  [Lock]: Harvester acquired lock for topic finding.")
                initial_topic = self.find_new_topic()
            print("  [Lock]: Harvester released lock after topic finding.")
            
            if not initial_topic:
                print("[Knowledge Harvester]: Could not find any new topic to learn about. Ending cycle.")
                break

            learned_topic = None
            
            wiki_result = self.get_fact_from_wikipedia(initial_topic)
            if wiki_result:
                final_topic, fact_sentence = wiki_result
                if self._try_to_learn(fact_sentence):
                    learned_this_interval = True
                    learned_topic = final_topic
            
            if not learned_this_interval:
                time.sleep(1)
                ddg_result = self.get_fact_from_duckduckgo(initial_topic)
                if ddg_result:
                    final_topic, fact_sentence = ddg_result
                    if self._try_to_learn(fact_sentence):
                        learned_this_interval = True
                        learned_topic = final_topic

            if not learned_this_interval:
                print(f"[Knowledge Harvester]: Could not learn a fact for topic '{initial_topic}'. Rejecting for this session.")
                self.rejected_topics.add(self.agent._clean_phrase(initial_topic))
            else:
                if learned_topic and self.enable_anticipatory_cache:
                    self._anticipate_and_cache(learned_topic)
                break

        if not learned_this_interval:
            print("[Knowledge Harvester]: All learning attempts failed for this interval.")

        print("--- [Autonomous Learning Cycle Finished] ---\n")

    def _anticipate_and_cache(self, topic: str):
        print(f"\n  [Anticipatory Cache]: Pre-warming caches for the CORRECT topic: '{topic}'")
        try:
            simulated_question = f"what is {topic}"
            response_text = ""
            with self.lock:
                print("  [Lock]: Harvester acquired lock for anticipatory caching.")
                response_text = self.agent.chat(simulated_question)
            print("  [Lock]: Harvester released lock after caching.")

            failure_keywords = ["i'm not sure", "i am not sure", "rephrase", "i don't have any information", "i cannot answer", "uncertain"]
            if any(keyword in response_text.lower() for keyword in failure_keywords):
                print(f"  [Anticipatory Cache]: Pre-warming FAILED. Agent could not answer the simulated question. The cache was not confirmed.")
            else:
                print(f"  [Anticipatory Cache]: Caches for '{topic}' have been successfully pre-warmed.")

        except Exception as e:
            print(f"  [Anticipatory Cache Error]: Could not pre-warm cache for '{topic}'. Error: {e}")