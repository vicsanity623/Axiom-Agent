from __future__ import annotations

# knowledge_harvester.py
import random
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import quote

import requests
import wikipedia

if TYPE_CHECKING:
    from threading import Lock

    from axiom.cognitive_agent import CognitiveAgent

wikipedia.set_user_agent("AxiomAgent/1.0 (https://github.com/vicsanity623/Axiom-Agent)")


class LogColors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


class KnowledgeHarvester:
    __slots__ = ("agent", "lock", "rejected_topics")

    def __init__(self, agent: CognitiveAgent, lock: Lock) -> None:
        self.agent = agent
        self.lock = lock
        self.rejected_topics: set[str] = set()
        print("[Knowledge Harvester]: Initialized.")

    def discover_cycle(self) -> None:
        """
        The discovery cycle. Finds a single, new, unknown topic from a random source
        and triggers a learning goal for it.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- [Discovery Cycle Started at {timestamp}] ---")

        new_topic = self._find_new_topic()

        if new_topic:
            goal = f"INVESTIGATE: {new_topic}"
            with self.lock:
                if goal not in self.agent.learning_goals:
                    self.agent.learning_goals.append(goal)
                    print(
                        f"  [Discovery]: Found new topic '{new_topic}'. Added to learning goals.",
                    )
        else:
            print(
                f"[Discovery Cycle]: {LogColors.YELLOW}Could not find any new topics to learn about this cycle.{LogColors.RESET}",
            )

        print("--- [Discovery Cycle Finished] ---\n")
        self.agent.log_autonomous_cycle_completion()

    def _resolve_investigation_goal(self, goal: str) -> bool:
        """
        Takes a learning goal and tries to find and learn its definition,
        prioritizing the Dictionary API.
        """
        match = re.match(r"INVESTIGATE: (.*)", goal)
        if not match:
            return False
        word_to_learn = match.group(1).lower()

        print(
            f"[Study Cycle]: Prioritizing learning goal: To define '{word_to_learn}'.",
        )

        api_result = self.get_definition_from_api(word_to_learn)

        if api_result:
            part_of_speech, definition = api_result
            with self.lock:
                self.agent.lexicon.add_linguistic_knowledge(
                    word=word_to_learn,
                    part_of_speech=part_of_speech,
                    definition=definition,
                )
                if goal in self.agent.learning_goals:
                    self.agent.learning_goals.remove(goal)
                self.agent.save_brain()
            return True

        print(
            f"  [Study Cycle]: Dictionary API failed for '{word_to_learn}'. Falling back to web search.",
        )

        queries = [f"define {word_to_learn}", f"what is a {word_to_learn}"]
        definition_found = None
        for query in queries:
            result = self.get_fact_from_wikipedia(
                query,
            ) or self.get_fact_from_duckduckgo(query)
            if result:
                definition_found = result[1]
                break
            time.sleep(1)

        if not definition_found:
            print(f"  [Study Cycle]: Web search also failed for '{word_to_learn}'.")
            return False

        pattern = re.compile(
            rf"(?i)\b{re.escape(word_to_learn)}\b\s*(\(.*\))?\s+(is|are|refers to)\s+((an?|the)?\s*([\w\s-]+))\b(.*)",
        )
        def_match = pattern.search(definition_found)
        if not def_match:
            print(
                f"  [Study Cycle]: Could not parse web search result: '{definition_found}'",
            )
            return False

        part_of_speech = def_match.group(5).strip().lower()
        full_definition = (def_match.group(3) + def_match.group(6)).strip()

        if part_of_speech.isdigit() or part_of_speech in ["a", "an", "the"]:
            return False

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

    def study_cycle(self) -> None:
        """
        The main study cycle. Prioritizes learning goals, then deepens existing knowledge.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- [Study Cycle Started at {timestamp}] ---")

        if self.agent.learning_goals:
            goal_to_resolve = self.agent.learning_goals[0]
            if goal_to_resolve.startswith("INVESTIGATE:"):
                goal_resolved = self._resolve_investigation_goal(goal_to_resolve)
                if not goal_resolved:
                    print(
                        f"  [Study Cycle]: {LogColors.YELLOW}Failed to resolve goal '{goal_to_resolve}'. Removing from queue.{LogColors.RESET}",
                    )
                    with self.lock:
                        if goal_to_resolve in self.agent.learning_goals:
                            self.agent.learning_goals.remove(goal_to_resolve)

                print(
                    f"{LogColors.GREEN}--- [Study Cycle Finished]: Completed a learning goal task. ---{LogColors.RESET}",
                )
                self.agent.log_autonomous_cycle_completion()
                return

        print(
            "[Study Cycle]: No learning goals. Attempting to deepen existing knowledge.",
        )
        self._deepen_knowledge_of_random_concept()

        print(f"{LogColors.GREEN}--- [Study Cycle Finished] ---\n{LogColors.RESET}")
        self.agent.log_autonomous_cycle_completion()

    def _deepen_knowledge_of_random_concept(self) -> None:
        """
        Picks a random node from the graph and tries to learn a new, related fact about it.
        This is the new, productive fallback for the study cycle.
        """
        stop_words = {
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "do",
            "a",
            "an",
            "the",
            "it",
            "they",
            "he",
            "she",
            "noun",
            "verb",
            "adjective",
            "article",
            "pronoun",
            "concept",
            "property",
        }

        random_node_name = None
        with self.lock:
            all_nodes = list(self.agent.graph.graph.nodes(data=True))
            if len(all_nodes) < 2:
                print(
                    "[Deepen Knowledge]: Not enough concepts in the brain to study yet.",
                )
                return

            _, node_data = random.choice(all_nodes)
            random_node_name = node_data.get("name")

        if not random_node_name or random_node_name in stop_words:
            if random_node_name:
                print(
                    f"[Deepen Knowledge]: Skipping study of common concept: '{random_node_name}'",
                )
            return

        print(f"[Deepen Knowledge]: Chosen to study the concept: '{random_node_name}'")

        result = self.get_fact_from_wikipedia(
            random_node_name,
        ) or self.get_fact_from_duckduckgo(random_node_name)

        if result:
            _title, fact_sentence = result

            with self.lock:
                print(
                    f"  [Deepen Knowledge]: {LogColors.GREEN}Attempting to learn new fact: '{fact_sentence}'{LogColors.RESET}",
                )
                self.agent.chat(fact_sentence)
        else:
            print(
                f"[Deepen Knowledge]: Could not find any new facts about '{random_node_name}'.",
            )

    def _find_new_topic(self, max_attempts: int = 5) -> str | None:
        """
        Finds a new, more focused, unknown topic using a curated list of subjects
        and a search result popularity heuristic.
        """
        core_subjects = [
            "Physics",
            "Chemistry",
            "Biology",
            "Mathematics",
            "Computer science",
            "History",
            "Geography",
            "Art",
            "Music",
            "Literature",
            "Philosophy",
            "Economics",
            "Psychology",
            "Sociology",
            "Astronomy",
            "Geology",
            "Common household items",
            "Types of animals",
            "Types of plants",
        ]

        for i in range(max_attempts):
            print(
                f"[Discovery]: Searching for a new topic (Attempt {i + 1}/{max_attempts})...",
            )
            try:
                subject = random.choice(core_subjects)
                print(f"  [Discovery]: Exploring the core subject: '{subject}'")

                related_topics = wikipedia.search(subject, results=10)
                if not related_topics:
                    continue

                topic = random.choice(related_topics)

                reject_keywords = ["list of", "timeline of", "index of", "outline of"]
                if any(keyword in topic.lower() for keyword in reject_keywords):
                    print(
                        f"  [Discovery Heuristic]: Rejecting meta-page topic: '{topic}'",
                    )
                    continue

                search_popularity = self._get_search_result_count(topic)
                minimum_popularity = 10000

                if (
                    search_popularity is not None
                    and search_popularity < minimum_popularity
                ):
                    print(
                        f"  [Discovery Heuristic]: Rejecting obscure topic '{topic}' (popularity: {search_popularity})",
                    )
                    continue

                clean_topic = self.agent._clean_phrase(topic)
                if (
                    not self.agent.lexicon.is_known_word(clean_topic)
                    and clean_topic not in self.rejected_topics
                ):
                    print(f"  [Discovery Success]: Found new, popular topic: '{topic}'")
                    return clean_topic

            except Exception as e:
                print(
                    f"  [Discovery Error]: An error occurred during topic finding. Error: {e}",
                )
                time.sleep(1)

        print(
            f"[Discovery Warning]: Could not find a new, suitable topic after {max_attempts} attempts.",
        )
        return None

    def _get_search_result_count(self, query: str) -> int | None:
        """
        Performs a DuckDuckGo search and scrapes the approximate result count.
        This is a heuristic for topic "popularity" or "commonness".
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
            }
            url = f"https://duckduckgo.com/html/?q={quote(query)}"

            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            match = re.search(r"([0-9,]+) results", response.text)
            if match:
                count_str = match.group(1).replace(",", "")
                return int(count_str)
        except Exception:
            return None
        return None

    def get_definition_from_api(self, word: str) -> tuple[str, str] | None:
        """
        Uses a free dictionary API to get a precise definition for a word.
        This is now the primary source for resolving 'INVESTIGATE' goals.
        Returns a tuple of (part_of_speech, definition_sentence).
        """
        print(f"[Knowledge Source]: Querying Dictionary API for '{word}'...")
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                print(f"  [Dictionary API]: Word '{word}' not found.")
                return None

            data = response.json()

            if not data or not isinstance(data, list):
                return None

            word_entry = data[0]
            if "meanings" in word_entry and word_entry["meanings"]:
                first_meaning = word_entry["meanings"][0]
                part_of_speech = first_meaning.get("partOfSpeech")

                if "definitions" in first_meaning and first_meaning["definitions"]:
                    first_definition = first_meaning["definitions"][0].get("definition")

                    if part_of_speech and first_definition:
                        article = (
                            "an"
                            if part_of_speech.lower().startswith(
                                ("a", "e", "i", "o", "u"),
                            )
                            else "a"
                        )
                        definition_sentence = f"{word} is {article} {part_of_speech}."

                        print(
                            f"  [Dictionary API]: Found definition: '{first_definition}'",
                        )
                        print(
                            f"  [Dictionary API]: Constructed fact: '{definition_sentence}'",
                        )

                        return (part_of_speech, first_definition)

        except requests.RequestException as e:
            print(f"  [Dictionary API]: An error occurred: {e}")
        except Exception:
            print(f"  [Dictionary API]: Failed to parse response for '{word}'.")

        return None

    def get_fact_from_wikipedia(self, topic: str) -> tuple[str, str] | None:
        """Gets the first simple sentence of a Wikipedia page."""
        print(f"[Knowledge Source]: Searching Wikipedia for '{topic}'...")
        try:
            search_results = wikipedia.search(topic, results=1)
            if not search_results:
                return None

            page = wikipedia.page(search_results[0], auto_suggest=False, redirect=True)

            if page and page.summary:
                first_sentence = page.summary.split(". ")[0].strip() + "."
                if self._is_sentence_simple_enough(first_sentence):
                    print(
                        f"  [Knowledge Source]: Extracted fact from Wikipedia: '{first_sentence}'",
                    )
                    return page.title, first_sentence
        except wikipedia.exceptions.DisambiguationError:
            print(
                f"  [Knowledge Source]: Wikipedia search for '{topic}' was ambiguous.",
            )
        except Exception:
            pass
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
                        f"  [Knowledge Source]: Extracted fact from DuckDuckGo: '{first_sentence}'",
                    )
                    return topic, first_sentence
        except Exception:
            pass
        return None

    def _is_sentence_simple_enough(
        self,
        sentence: str,
        max_words: int = 30,
        max_commas: int = 2,
    ) -> bool:
        """A simple filter to reject overly complex sentences."""
        return len(sentence.split()) <= max_words and sentence.count(",") <= max_commas
