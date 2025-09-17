# knowledge_harvester.py

import os
import random
import re
import time
from datetime import datetime

import requests
import wikipedia

from .graph_core import RelationshipEdge


class KnowledgeHarvester:
    def __init__(self, agent, lock):
        self.agent = agent
        self.lock = lock
        self.nyt_api_key = os.environ.get("NYT_API_KEY")
        if not self.nyt_api_key:
            print(
                "[Knowledge Harvester WARNING]: NYT_API_KEY environment variable not set. NYT topic source will be disabled.",
            )

        self.rejected_topics = set()
        self.enable_anticipatory_cache = False

        wikipedia.set_user_agent("AxiomAgent/1.0 (AxiomAgent@example.com)")
        print("[Knowledge Harvester]: Initialized.")

    def _pre_filter_headline(self, headline: str) -> str:
        print(f"  [Pre-filter]: Sanitizing raw headline: '{headline}'")
        main_part = re.split(r"[:|]", headline, 1)[0]
        main_part = re.sub(r"[^a-zA-Z0-9\s-]", "", main_part).strip()
        print(f"  [Pre-filter]: Simplified headline to '{main_part}' for LLM analysis.")
        return main_part

    def _extract_core_entity(self, topic_string: str) -> str | None:
        if len(topic_string) < 10 or len(topic_string) > 250:
            print(
                f"  [Guardrail]: Topic rejected (invalid length). Topic: '{topic_string}'",
            )
            return None

        print(
            f"  [Harvester]: Analyzing topic string for core entity: '{topic_string}'",
        )
        try:
            interpretation = self.agent.interpreter.interpret(topic_string)
            entities = interpretation.get("entities", [])
            if entities:
                priority_types = [
                    "PERSON",
                    "ORGANIZATION",
                    "LOCATION",
                    "GPE",
                    "PRODUCT",
                    "EVENT",
                ]
                for ent_type in priority_types:
                    for entity in entities:
                        if entity.get("type") == ent_type:
                            core_entity = entity.get("name")
                            print(
                                f"  [Harvester]: Extracted core entity: '{core_entity}'",
                            )
                            return core_entity
                core_entity = entities[0].get("name")
                print(
                    f"  [Harvester]: Extracted core entity (fallback): '{core_entity}'",
                )
                return core_entity
        except Exception as e:
            print(f"  [Harvester]: Error during core entity extraction: {e}")
        return None

    def get_archival_topic(self) -> str | None:
        if not self.nyt_api_key:
            return None
        print(
            "  [Discovery]: Attempting to fetch a random topic from New York Times Archives...",
        )
        try:
            current_year = datetime.now().year
            year = random.randint(2000, current_year)
            month = random.randint(1, 12)
            print(f"    - Searching archive for {year}-{month:02d}...")
            url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={self.nyt_api_key}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            if not docs:
                return None
            random.shuffle(docs)
            for article in docs:
                headline = article.get("headline", {}).get("main", "")
                if headline and len(headline.split()) > 3:
                    safe_topic_string = self._pre_filter_headline(headline)
                    topic = self._extract_core_entity(safe_topic_string)
                    if topic:
                        return topic
        except requests.RequestException as e:
            print(f"  [Discovery]: Error fetching NYT Archive API: {e}")
        return None

    def _is_sentence_simple_enough(
        self,
        sentence: str,
        max_words=40,
        max_commas=3,
    ) -> bool:
        if len(sentence.split()) > max_words:
            print(
                f"  [Simplicity Filter]: Sentence rejected (too long: {len(sentence.split())} words).",
            )
            return False
        if sentence.count(",") > max_commas:
            print(
                f"  [Simplicity Filter]: Sentence rejected (too complex: {sentence.count(',')} commas).",
            )
            return False
        return True

    def _try_to_learn(self, fact_sentence: str) -> bool:
        if not fact_sentence:
            return False
        with self.lock:
            print("  [Lock]: Harvester acquired lock.")
            learned_successfully = self.agent.learn_new_fact_autonomously(fact_sentence)
        print("  [Lock]: Harvester released lock.")
        return learned_successfully

    def discover_new_topic_and_learn(self):
        print("\n--- [Discovery Cycle Started] ---")
        initial_topic = None
        with self.lock:
            print("  [Lock]: Harvester acquired lock for topic finding.")
            initial_topic = self._find_new_topic()
        print("  [Lock]: Harvester released lock after topic finding.")

        if not initial_topic:
            print(
                "[Discovery Cycle]: Could not find any new topic to learn about. Ending cycle.",
            )
            print("--- [Discovery Cycle Finished] ---\n")
            self.agent.log_autonomous_cycle_completion()  # Log completion for reboot logic
            return

        learned_topic = None
        wiki_result = self.get_fact_from_wikipedia(initial_topic)
        if wiki_result:
            final_topic, fact_sentence = wiki_result
            if self._try_to_learn(fact_sentence):
                learned_topic = final_topic

        if not learned_topic:
            time.sleep(1)
            ddg_result = self.get_fact_from_duckduckgo(initial_topic)
            if ddg_result:
                final_topic, fact_sentence = ddg_result
                if self._try_to_learn(fact_sentence):
                    learned_topic = final_topic

        if not learned_topic:
            print(
                f"[Discovery Cycle]: Could not learn a fact for topic '{initial_topic}'. Rejecting for this session.",
            )
            self.rejected_topics.add(self.agent._clean_phrase(initial_topic))
        else:
            if self.enable_anticipatory_cache:
                self._anticipate_and_cache(learned_topic)

        print("--- [Discovery Cycle Finished] ---\n")
        self.agent.log_autonomous_cycle_completion()  # Log completion for reboot logic

    def _find_new_topic(self, max_attempts=10) -> str | None:
        for i in range(max_attempts):
            print(
                f"\n[Discovery]: Searching for a new topic (Attempt {i + 1}/{max_attempts})...",
            )
            topic = None
            source_choice = random.choice(["nyt", "wiki", "wiki"])

            if source_choice == "nyt" and self.nyt_api_key:
                topic = self.get_archival_topic()
            else:
                topic = self.get_random_wikipedia_topic()

            if topic:
                clean_topic = self.agent._clean_phrase(topic)
                if self.agent.graph.get_node_by_name(clean_topic):
                    print(
                        f"  [Discovery]: Agent already knows about '{topic}'. Finding a new topic...",
                    )
                    continue
                if clean_topic in self.rejected_topics:
                    print(
                        f"  [Discovery]: Agent has already tried and failed to learn about '{topic}'. Finding a new topic...",
                    )
                    continue
                print(
                    f"  [Discovery Success]: Found a new, unknown topic: '{topic}'",
                )
                return topic

        print(
            f"[Discovery Warning]: Could not find a new, unknown topic after {max_attempts} attempts.",
        )
        return None

    def get_random_wikipedia_topic(self) -> str | None:
        import wikipediaapi

        wiki_api = wikipediaapi.Wikipedia(
            "AxiomAgent/1.0 (AxiomAgent@example.com)",
            "en",
        )
        print(
            "  [Discovery]: Attempting to find a random topic from Wikipedia categories...",
        )
        major_categories = [
            "History",
            "Science",
            "Technology",
            "Art",
            "Geography",
            "Mathematics",
            "Philosophy",
            "Culture",
            "Health",
            "Nature",
            "People",
            "Society",
            "Learning",
            "Knowledge",
            "Curriculum",
        ]
        try:
            category_name = random.choice(major_categories)
            category_page = wiki_api.page(f"Category:{category_name}")
            if category_page.exists() and hasattr(category_page, "categorymembers"):
                members = list(category_page.categorymembers.values())
                pages = [m for m in members if m.ns == wikipediaapi.Namespace.MAIN]
                if pages:
                    chosen_page = random.choice(pages)
                    if not any(
                        keyword in chosen_page.title.lower()
                        for keyword in [
                            "list of",
                            "index of",
                            "timeline of",
                            "outline of",
                        ]
                    ):
                        return chosen_page.title
        except Exception as e:
            print(f"  [Discovery]: Error during Wikipedia search: {e}")
        return None

    def study_existing_concept(self):
        print("\n--- [Study Cycle Started] ---")

        chosen_edge = None
        source_node = None
        target_node = None

        with self.lock:
            # --- THIS IS THE FIX ---
            # We now iterate through (u, v, data) to get the complete edge info.
            # Then, we build a full dictionary for our from_dict method.
            all_edges_data = []
            for u, v, data in self.agent.graph.graph.edges(data=True):
                full_data = data.copy()  # Start with the existing edge attributes
                full_data["source"] = u  # Add the source ID
                full_data["target"] = v  # Add the target ID
                all_edges_data.append(full_data)

            if not all_edges_data:
                print("[Study Cycle]: Brain has no facts to study yet.")
                print("--- [Study Cycle Finished] ---\n")
                self.agent.log_autonomous_cycle_completion()
                return

            # Now, we can correctly reconstruct the full RelationshipEdge objects
            all_edges = [RelationshipEdge.from_dict(d) for d in all_edges_data]
            chosen_edge = random.choice(all_edges)

            # This logic will now succeed because chosen_edge.source has a valid ID
            source_node_data = self.agent.graph.graph.nodes.get(chosen_edge.source)
            target_node_data = self.agent.graph.graph.nodes.get(chosen_edge.target)

            if source_node_data:
                source_node = self.agent.graph.get_node_by_name(
                    source_node_data["name"],
                )
            if target_node_data:
                target_node = self.agent.graph.get_node_by_name(
                    target_node_data["name"],
                )

        if not source_node or not target_node:
            # This message should no longer appear, but it's good to keep as a safeguard
            print(
                "[Study Cycle]: Could not retrieve nodes for a chosen fact. Ending cycle.",
            )
            print("--- [Study Cycle Finished] ---\n")
            self.agent.log_autonomous_cycle_completion()
            return

        study_topic = source_node.name
        known_fact = f"{source_node.name} {chosen_edge.type.replace('_', ' ')} {target_node.name}."
        print(
            f"[Study Cycle]: Chosen to study '{study_topic}' based on the known fact: '{known_fact}'",
        )

        questions = self._generate_study_questions(study_topic, known_fact)
        if not questions:
            print("[Study Cycle]: Could not generate study questions. Ending cycle.")
            print("--- [Study Cycle Finished] ---\n")
            self.agent.log_autonomous_cycle_completion()
            return

        learned_something_new = False
        for question in questions:
            print(
                f"\n[Study Cycle]: Seeking answer for self-generated question: '{question}'",
            )
            # This part of your logic remains the same
            wiki_result = self.get_fact_from_wikipedia(question)
            if wiki_result and self._try_to_learn(wiki_result[1]):
                learned_something_new = True
                continue

            time.sleep(1)
            ddg_result = self.get_fact_from_duckduckgo(question)
            if ddg_result and self._try_to_learn(ddg_result[1]):
                learned_something_new = True

        if not learned_something_new:
            print(
                "[Study Cycle]: Completed study, but did not find any new, simple facts to learn.",
            )

        print("--- [Study Cycle Finished] ---\n")
        self.agent.log_autonomous_cycle_completion()

    def _generate_study_questions(self, topic: str, known_fact: str) -> list[str]:
        print(
            f"  [Study Engine]: Generating curious follow-up questions for '{topic}'...",
        )
        try:
            return self.agent.interpreter.generate_curious_questions(topic, known_fact)
        except Exception as e:
            print(f"  [Study Engine]: Error during question generation: {e}")
            return []

    def get_fact_from_wikipedia(self, topic: str) -> tuple[str, str] | None:
        print(
            f"[Knowledge Source]: Performing intelligent search for '{topic}' on Wikipedia...",
        )
        try:
            search_results = wikipedia.search(topic, results=5)
            if not search_results:
                return None

            page_title = search_results[0]
            page = wikipedia.page(page_title, auto_suggest=False, redirect=True)

            if page and page.summary:
                first_sentence = page.summary.split(". ")[0].strip()
                if not first_sentence.endswith("."):
                    first_sentence += "."

                if len(first_sentence.split()) > 5 and self._is_sentence_simple_enough(
                    first_sentence,
                ):
                    print(
                        f"  [Knowledge Source]: Extracted fact from Wikipedia: '{first_sentence}'",
                    )
                    return page.title, first_sentence
        except Exception:
            return None
        return None

    def get_fact_from_duckduckgo(self, topic: str) -> tuple[str, str] | None:
        print(
            f"[Knowledge Source]: Falling back to DuckDuckGo with query: '{topic}'...",
        )
        try:
            url = f"https://api.duckduckgo.com/?q={topic}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            definition = data.get("AbstractText") or data.get("Definition")
            if definition:
                fact_sentence = definition.split(". ")[0].strip()
                if not fact_sentence.endswith("."):
                    fact_sentence += "."
                if "is a redirect to" in fact_sentence.lower():
                    return None

                if self._is_sentence_simple_enough(fact_sentence):
                    print(
                        f"  [Knowledge Source]: Extracted fact from DuckDuckGo: '{fact_sentence}'",
                    )
                    return topic, fact_sentence
        except Exception:
            return None
        return None

    def _anticipate_and_cache(self, topic: str):
        print(
            f"\n  [Anticipatory Cache]: Pre-warming caches for the CORRECT topic: '{topic}'",
        )
        try:
            simulated_question = f"what is {topic}"
            response_text = ""
            with self.lock:
                print("  [Lock]: Harvester acquired lock for anticipatory caching.")
                response_text = self.agent.chat(simulated_question)
            print("  [Lock]: Harvester released lock after caching.")
            failure_keywords = [
                "i'm not sure",
                "i am not sure",
                "rephrase",
                "i don't have any information",
                "i cannot answer",
                "uncertain",
            ]
            if any(keyword in response_text.lower() for keyword in failure_keywords):
                print(
                    "  [Anticipatory Cache]: Pre-warming FAILED. Agent could not answer the simulated question.",
                )
            else:
                print(
                    f"  [Anticipatory Cache]: Caches for '{topic}' have been successfully pre-warmed.",
                )
        except Exception as e:
            print(
                f"  [Anticipatory Cache Error]: Could not pre-warm cache for '{topic}'. Error: {e}",
            )
