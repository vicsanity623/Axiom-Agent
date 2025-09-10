# knowledge_harvester.py

import time
import random
import requests
import wikipediaapi
import os
import json

class KnowledgeHarvester:
    def __init__(self, agent, lock):
        """
        Initializes the harvester with a reference to the cognitive agent and a lock.
        """
        self.agent = agent
        self.lock = lock
        self.nyt_api_key = os.environ.get("NYT_API_KEY")
        if not self.nyt_api_key:
            print("[Knowledge Harvester WARNING]: NYT_API_KEY environment variable not set. Trending topic source will be disabled.")
        self.wiki_wiki = wikipediaapi.Wikipedia('AxiomAgent/1.0 (AxiomAgent@example.com)', 'en')
        print("[Knowledge Harvester]: Initialized.")

    def _extract_core_entity(self, topic_string: str) -> str | None:
        """Uses the agent's own interpreter to find the main entity in a string."""
        print(f"  [Harvester]: Analyzing topic string for core entity: '{topic_string}'")
        try:
            interpretation = self.agent.interpreter.interpret(topic_string)
            entities = interpretation.get('entities', [])
            if entities:
                # Prioritize specific entity types over miscellaneous ones
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
        """
        Finds a trending or random topic that the agent does not already have in its brain.
        """
        for i in range(max_attempts):
            print(f"\n[Harvester]: Searching for a new topic (Attempt {i + 1}/{max_attempts})...")
            topic = None
            if i % 2 == 0 and self.nyt_api_key:
                topic = self.get_trending_topic()
            else:
                topic = self.get_random_wikipedia_topic()
            
            if topic:
                if self.agent.graph.get_node_by_name(self.agent._clean_phrase(topic)):
                    print(f"  [Harvester Info]: Agent already knows about '{topic}'. Finding a new topic...")
                    time.sleep(1)
                    continue
                else:
                    print(f"  [Harvester Success]: Found a new, unknown topic: '{topic}'")
                    return topic
        
        print(f"[Harvester Warning]: Could not find a new, unknown topic after {max_attempts} attempts.")
        return None

    def get_trending_topic(self) -> str | None:
        if not self.nyt_api_key: return None
        print("  [Harvester]: Attempting to fetch trending topics from New York Times API...")
        try:
            url = f"https://api.nytimes.com/svc/mostpopular/v2/viewed/1.json?api-key={self.nyt_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'OK' and data.get('results'):
                articles = data['results']
                potential_topics = []
                for article in articles[:10]:
                    keywords_str = article.get('adx_keywords', '')
                    if ';' in keywords_str:
                        main_keyword = keywords_str.split(';')[0]
                        if '(' in main_keyword: main_keyword = main_keyword.split('(')[0].strip()
                        if main_keyword and len(main_keyword) > 3 and main_keyword.lower() not in ['news', 'politics', 'business']:
                             potential_topics.append(main_keyword)
                if potential_topics:
                    chosen_topic_string = random.choice(potential_topics)
                    return self._extract_core_entity(chosen_topic_string)
        except requests.RequestException as e:
            print(f"  [Harvester]: Error fetching NYT API: {e}")
        return None

    def get_random_wikipedia_topic(self) -> str | None:
        print("  [Harvester]: Attempting to find a random topic from Wikipedia...")
        major_categories = ["History", "Science", "Technology", "Art", "Geography", "Mathematics", "Philosophy", "Culture", "Health", "Nature", "People", "Society"]
        try:
            category_name = random.choice(major_categories)
            category_page = self.wiki_wiki.page(f"Category:{category_name}")
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

    def get_fact_from_wikipedia(self, topic: str) -> str | None:
        print(f"[Knowledge Harvester]: Attempting to find fact for '{topic}' on Wikipedia...")
        page = self.wiki_wiki.page(topic)
        if page.exists() and page.summary:
            first_sentence = page.summary.split('. ')[0].strip()
            if not first_sentence.endswith('.'): first_sentence += '.'
            if len(first_sentence.split()) > 5:
                print(f"[Knowledge Harvester]: Extracted fact from Wikipedia: '{first_sentence}'")
                return first_sentence
        print(f"[Knowledge Harvester]: Could not find a suitable fact for '{topic}' on Wikipedia.")
        return None

    def get_fact_from_duckduckgo(self, topic: str) -> str | None:
        print(f"[Knowledge Harvester]: Wikipedia fact was unsuitable. Falling back to DuckDuckGo for '{topic}'...")
        try:
            url = f"https://api.duckduckgo.com/?q={topic}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            definition = data.get('AbstractText') or data.get('Definition')
            if definition:
                fact_sentence = definition.split('. ')[0].strip()
                if not fact_sentence.endswith('.'): fact_sentence += '.'
                if "is a redirect to" in fact_sentence.lower(): return None
                print(f"[Knowledge Harvester]: Extracted fact from DuckDuckGo: '{fact_sentence}'")
                return fact_sentence
        except requests.RequestException as e:
            print(f"[Knowledge Harvester]: Error fetching from DuckDuckGo API: {e}")
        except json.JSONDecodeError:
            print("[Knowledge Harvester]: Error decoding JSON from DuckDuckGo API.")
        return None

    def _anticipate_and_cache(self, topic: str):
        """
        After learning about a new topic, this method simulates a user asking
        about it to pre-warm the interpreter and synthesizer caches.
        """
        print(f"\n  [Anticipatory Cache]: Pre-warming caches for topic: '{topic}'")
        try:
            # The most likely question a user will ask is "who is/what is [topic]?"
            simulated_question = f"what is {topic}"
            
            # This call will be slow, but it's happening in the background during idle time.
            # It will force the agent to interpret the question, query its brain for the
            # new fact, and synthesize a fluent response.
            self.agent.chat(simulated_question)
            
            print(f"  [Anticipatory Cache]: Caches for '{topic}' have been successfully pre-warmed.")
        except Exception as e:
            print(f"  [Anticipatory Cache Error]: Could not pre-warm cache for '{topic}'. Error: {e}")

    def _try_to_learn(self, fact_sentence: str) -> bool:
        """
        A helper function that attempts to learn a single fact and returns True on success.
        """
        if not fact_sentence:
            return False
            
        with self.lock:
            print("  [Lock]: Harvester acquired lock.")
            learned_successfully = self.agent.learn_new_fact_autonomously(fact_sentence)
        print("  [Lock]: Harvester released lock.")
        return learned_successfully

    def harvest_and_learn(self):
        """
        The main autonomous function. Now includes anticipatory caching.
        """
        print("\n--- [Autonomous Learning Cycle Started] ---")
        
        topic = None
        with self.lock:
            print("  [Lock]: Harvester acquired lock for topic finding.")
            topic = self.find_new_topic()
        print("  [Lock]: Harvester released lock after topic finding.")
            
        if topic:
            learned_something = False
            fact_from_wiki = self.get_fact_from_wikipedia(topic)
            if fact_from_wiki:
                learned_something = self._try_to_learn(fact_from_wiki)
            
            if not learned_something:
                time.sleep(1)
                fact_from_ddg = self.get_fact_from_duckduckgo(topic)
                if fact_from_ddg:
                    learned_something = self._try_to_learn(fact_from_ddg)
            
            # --- NEW: If learning was successful, trigger anticipatory caching ---
            if learned_something:
                with self.lock:
                    print("  [Lock]: Harvester acquired lock for anticipatory caching.")
                    self._anticipate_and_cache(topic)
                print("  [Lock]: Harvester released lock after caching.")
            else:
                print("[Knowledge Harvester]: All sources attempted, but could not learn a simple fact for this topic.")
        else:
            print("[Knowledge Harvester]: Could not find any new topic to learn about in this cycle.")
            
        print("--- [Autonomous Learning Cycle Finished] ---\n")