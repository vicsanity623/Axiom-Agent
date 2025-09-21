from __future__ import annotations

# knowledge_harvester.py
import random
import re
import time
from typing import TYPE_CHECKING

import requests
import wikipedia

if TYPE_CHECKING:
    from threading import Lock

    from axiom.cognitive_agent import CognitiveAgent

# Set a user agent for Wikipedia API requests, which is good practice.
wikipedia.set_user_agent("AxiomAgent/1.0 (https://github.com/vicsanity623/Axiom-Agent)")


class KnowledgeHarvester:
    __slots__ = ("agent", "lock", "rejected_topics")

    def __init__(self, agent: CognitiveAgent, lock: Lock) -> None:
        self.agent = agent
        self.lock = lock
        self.rejected_topics: set[str] = set()
        print("[Knowledge Harvester]: Initialized.")

    # --- CORE AUTONOMOUS CYCLES ---

    def study_cycle(self) -> None:
        """
        The main study cycle. It prioritizes resolving 'INVESTIGATE' learning goals first.
        If no goals exist, it falls back to studying an existing concept to find new connections.
        """
        print("\n--- [Study Cycle Started] ---")

        goal_resolved = False
        if self.agent.learning_goals:
            # Get the oldest goal first (First-In, First-Out)
            goal_to_resolve = self.agent.learning_goals[0]
            if goal_to_resolve.startswith("INVESTIGATE:"):
                goal_resolved = self._resolve_investigation_goal(goal_to_resolve)

        if goal_resolved:
            print(
                "--- [Study Cycle Finished]: Successfully resolved a learning goal. ---"
            )
            self.agent.log_autonomous_cycle_completion()
            return

        # If no goals were resolved, fall back to the old "study a random fact" logic
        self._study_random_concept()

        print("--- [Study Cycle Finished] ---\n")
        self.agent.log_autonomous_cycle_completion()

    def discover_cycle(self) -> None:
        """
        The discovery cycle. Finds a single, new, unknown topic from a random source
        and triggers a learning goal for it.
        """
        print("\n--- [Discovery Cycle Started] ---")

        # This logic is now much simpler. Its only job is to find a word we don't know.
        new_topic = self._find_new_topic()

        if new_topic:
            # Instead of trying to learn a fact, we now create a learning goal
            # to define the topic itself. This is a more robust approach.
            goal = f"INVESTIGATE: {new_topic}"
            with self.lock:
                if goal not in self.agent.learning_goals:
                    self.agent.learning_goals.append(goal)
                    print(
                        f"  [Discovery]: Found new topic '{new_topic}'. Added to learning goals."
                    )
        else:
            print(
                "[Discovery Cycle]: Could not find any new topics to learn about this cycle."
            )

        print("--- [Discovery Cycle Finished] ---\n")
        self.agent.log_autonomous_cycle_completion()

    # --- NEW: GOAL-DRIVEN RESEARCH METHOD ---

    def _resolve_investigation_goal(self, goal: str) -> bool:
        """
        Takes a learning goal (e.g., "INVESTIGATE: platypus") and tries to
        find and learn its definition from the web using deterministic methods.
        """
        match = re.match(r"INVESTIGATE: (\w+)", goal)
        if not match:
            return False
        word_to_learn = match.group(1).lower()

        print(
            f"[Study Cycle]: Prioritizing learning goal: To define '{word_to_learn}'."
        )

        # Formulate targeted search queries for definitions
        queries = [
            f"define {word_to_learn}",
            f"what is a {word_to_learn}",
            f"meaning of {word_to_learn}",
        ]

        definition_found = None
        for query in queries:
            result = self.get_fact_from_wikipedia(
                query
            ) or self.get_fact_from_duckduckgo(query)
            if result:
                definition_found = result[1]  # result is a tuple (title, sentence)
                break
            time.sleep(1)

        if not definition_found:
            print(
                f"  [Study Cycle]: Could not find a simple definition for '{word_to_learn}'."
            )
            return False

        # Use Regular Expressions to parse the definition (No NLP/Spacy!)
        pattern = re.compile(
            rf"(?i)\b{word_to_learn}\b\s+(is|are|refers to)\s+(an?|the)?\s*(\w+)\b(.*)"
        )
        def_match = pattern.search(definition_found)

        if not def_match:
            print(
                f"  [Study Cycle]: Found sentence, but could not parse it into a definition: '{definition_found}'"
            )
            return False

        part_of_speech = def_match.group(3).lower()
        full_definition = (def_match.group(3) + def_match.group(4)).strip()

        print(f"  [Study Cycle]: Successfully parsed definition for '{word_to_learn}'.")

        with self.lock:
            self.agent.lexicon.add_linguistic_knowledge(
                word=word_to_learn,
                part_of_speech=part_of_speech,
                definition=full_definition,
            )
            if goal in self.agent.learning_goals:
                self.agent.learning_goals.remove(goal)
            self.agent.save_brain()

        return True

    # --- HELPER & LEGACY METHODS ---

    def _study_random_concept(self) -> None:
        """
        Legacy study method. Picks a random fact and tries to find a related one.
        This no longer uses an LLM to generate questions.
        """
        print("[Study Cycle]: No learning goals. Studying a random existing concept.")

        source_node_name, target_node_name = None, None
        with self.lock:
            all_edges = list(self.agent.graph.graph.edges(data=True))
            if not all_edges:
                print("[Study Cycle]: Brain has no facts to study yet.")
                return

            # Choose a random fact to start from
            u, v, _ = random.choice(all_edges)
            source_node_data = self.agent.graph.graph.nodes.get(u)
            target_node_data = self.agent.graph.graph.nodes.get(v)
            if source_node_data and target_node_data:
                source_node_name = source_node_data["name"]
                target_node_name = target_node_data["name"]

        if source_node_name and target_node_name:
            # We will now use the TARGET of the fact as the new research topic
            study_topic = target_node_name
            print(
                f"[Study Cycle]: Studying connection from '{source_node_name}' to '{study_topic}'."
            )

            # Instead of asking a question, we create an investigation goal
            goal = f"INVESTIGATE: {study_topic.lower().split()[0]}"  # Use first word of topic
            with self.lock:
                if (
                    goal not in self.agent.learning_goals
                    and not self.agent.lexicon.is_known_word(study_topic)
                ):
                    self.agent.learning_goals.append(goal)
                    print(
                        f"  [Study Cycle]: Curiosity triggered. Added goal to investigate '{study_topic}'."
                    )

    def _find_new_topic(self, max_attempts: int = 5) -> str | None:
        """Finds a new, unknown topic from Wikipedia."""
        for i in range(max_attempts):
            print(
                f"[Discovery]: Searching for a new topic (Attempt {i + 1}/{max_attempts})..."
            )

            try:
                # wikipedia.random() is the simplest way to get a new topic
                topic = wikipedia.random(pages=1)
                clean_topic = self.agent._clean_phrase(topic)

                # Check if we know it or have rejected it
                if (
                    not self.agent.lexicon.is_known_word(clean_topic)
                    and clean_topic not in self.rejected_topics
                ):
                    print(f"  [Discovery Success]: Found new, unknown topic: '{topic}'")
                    return clean_topic
            except Exception as e:
                print(
                    f"  [Discovery Error]: Could not fetch random topic from Wikipedia. Error: {e}"
                )
                time.sleep(1)  # Wait before retrying

        print(
            f"[Discovery Warning]: Could not find a new, unknown topic after {max_attempts} attempts."
        )
        return None

    def get_fact_from_wikipedia(self, topic: str) -> tuple[str, str] | None:
        """Gets the first simple sentence of a Wikipedia page."""
        print(f"[Knowledge Source]: Searching Wikipedia for '{topic}'...")
        try:
            # Use search to handle ambiguity, then get the top result page
            search_results = wikipedia.search(topic, results=1)
            if not search_results:
                return None

            page = wikipedia.page(search_results[0], auto_suggest=False, redirect=True)

            if page and page.summary:
                first_sentence = page.summary.split(". ")[0].strip() + "."
                if self._is_sentence_simple_enough(first_sentence):
                    print(
                        f"  [Knowledge Source]: Extracted fact from Wikipedia: '{first_sentence}'"
                    )
                    return page.title, first_sentence
        except wikipedia.exceptions.DisambiguationError:
            print(
                f"  [Knowledge Source]: Wikipedia search for '{topic}' was ambiguous."
            )
        except Exception:
            pass  # Suppress other common wikipedia library errors
        return None

    def get_fact_from_duckduckgo(self, topic: str) -> tuple[str, str] | None:
        """Gets a definition from DuckDuckGo's Instant Answer API."""
        print(f"[Knowledge Source]: Searching DuckDuckGo for '{topic}'...")
        try:
            url = f"https://api.duckduckgo.com/?q={topic}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            definition = data.get("AbstractText") or data.get("Definition")
            if definition:
                first_sentence = definition.split(". ")[0].strip() + "."
                if self._is_sentence_simple_enough(first_sentence):
                    print(
                        f"  [Knowledge Source]: Extracted fact from DuckDuckGo: '{first_sentence}'"
                    )
                    return topic, first_sentence
        except Exception:
            pass
        return None

    def _is_sentence_simple_enough(
        self, sentence: str, max_words: int = 30, max_commas: int = 2
    ) -> bool:
        """A simple filter to reject overly complex sentences."""
        return len(sentence.split()) <= max_words and sentence.count(",") <= max_commas
