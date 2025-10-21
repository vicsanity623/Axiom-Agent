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
from nltk.stem import WordNetLemmatizer

if TYPE_CHECKING:
    from threading import Lock

    from axiom.cognitive_agent import CognitiveAgent
    from axiom.graph_core import ConceptNode, RelationshipEdge

wikipedia.set_user_agent("AxiomAgent/1.0 (https://github.com/vicsanity623/Axiom-Agent)")


class LogColors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


class KnowledgeHarvester:
    __slots__ = ("agent", "lock", "rejected_topics", "lemmatizer")

    def __init__(self, agent: CognitiveAgent, lock: Lock) -> None:
        """Initialize the KnowledgeHarvester.

        Args:
            agent: The instance of the CognitiveAgent this harvester will serve.
            lock: A threading lock to ensure thread-safe operations on the agent.
        """
        self.agent = agent
        self.lock = lock
        self.lemmatizer = WordNetLemmatizer()
        self.rejected_topics: set[str] = set()
        print("[Knowledge Harvester]: Initialized.")

    def discover_cycle(self) -> None:
        """Run one full discovery cycle to find a new topic to learn.

        This cycle uses an intelligent search strategy to find a new,
        relevant, and popular topic that is not already in the agent's
        lexicon. If a suitable topic is found, it creates a new
        "INVESTIGATE" goal and adds it to the agent's learning queue.
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
        """Resolve an "INVESTIGATE" goal by finding a word's definition."""
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
                self.agent.lexicon.add_linguistic_knowledge_quietly(
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

        queries = [f"what is {word_to_learn}", f"define {word_to_learn}", word_to_learn]
        web_fact = None
        source_topic = None
        for query in queries:
            result = self.get_fact_from_wikipedia(
                query,
            ) or self.get_fact_from_duckduckgo(query)
            if result:
                source_topic, web_fact = result
                break
            time.sleep(1)

        if not web_fact:
            print(f"  [Study Cycle]: Web search also failed for '{word_to_learn}'.")
            return False

        print(f"  [Study Cycle]: Found potential fact from web: '{web_fact}'")
        with self.lock:
            was_learned = self.agent.learn_new_fact_autonomously(
                fact_sentence=web_fact,
                source_topic=source_topic or word_to_learn,
            )

        if was_learned:
            print("  [Study Cycle]: Agent successfully learned the new fact.")
            with self.lock:
                if goal in self.agent.learning_goals:
                    self.agent.learning_goals.remove(goal)
            return True
        print("  [Study Cycle]: Agent failed to learn from the web fact.")
        return False

    def study_cycle(self) -> None:
        """Run one full study cycle.

        This cycle has two main priorities:
        1.  **Resolve Goals:** It first checks for any pending "INVESTIGATE"
            goals in the agent's learning queue and attempts to resolve them.
        2.  **Deepen Knowledge:** If the queue is empty, it falls back to the
            `_deepen_knowledge_of_random_concept` routine to proactively
            enrich the agent's existing knowledge.
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

    def refinement_cycle(self) -> None:
        """Run one full introspection and refinement cycle.

        This cycle allows the agent to improve the quality of its own knowledge.
        It searches for "chunky" facts (e.g., long, definitional concepts)
        and uses the LLM to break them down into smaller, more precise, atomic
        facts. This improves the agent's ability to reason symbolically.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- [Refinement Cycle Started at {timestamp}] ---")

        chunky_fact = None
        with self.lock:
            chunky_fact = self._find_chunky_fact()

        if chunky_fact:
            source_node, target_node, edge = chunky_fact
            print(
                f"  [Refinement]: Found a chunky fact to refine: '{source_node.name}' --[{edge.type}]--> '{target_node.name}'",
            )

            atomic_sentences = self.agent.interpreter.break_down_definition(
                subject=source_node.name,
                chunky_definition=target_node.name,
            )

            if atomic_sentences:
                print(
                    f"  [Refinement]: Decomposed into {len(atomic_sentences)} new atomic facts.",
                )
                for sentence in atomic_sentences:
                    with self.lock:
                        self.agent.learn_new_fact_autonomously(sentence)

                with self.lock:
                    graph_edge = self.agent.graph.graph[edge.source][edge.target]
                    key_to_modify = None
                    for key, data in graph_edge.items():
                        if data.get("type") == edge.type:
                            key_to_modify = key
                            break

                    if key_to_modify is not None:
                        self.agent.graph.graph[edge.source][edge.target][key_to_modify][
                            "weight"
                        ] = 0.2
                        self.agent.save_brain()
                        print(
                            "  [Refinement]: Marked original fact as refined by lowering its weight.",
                        )

        else:
            print("[Refinement]: No chunky facts found for refinement this cycle.")

        print(
            f"{LogColors.GREEN}--- [Refinement Cycle Finished] ---\n{LogColors.RESET}",
        )
        self.agent.log_autonomous_cycle_completion()

    def _find_chunky_fact(
        self,
    ) -> tuple[ConceptNode, ConceptNode, RelationshipEdge] | None:
        """Find a single, high-confidence, unrefined "chunky" fact.

        A "chunky" fact is defined as a relationship where the object is a
        long noun phrase, suggesting it's a definition that could be
        broken down into smaller, more atomic facts.

        This method searches for `is_a` relationships with long targets.

        Returns:
            A tuple containing the source node, target node, and edge object
            of a suitable fact, or None if none is found.
        """
        potential_facts = []

        all_edges = self.agent.graph.get_all_edges()

        for edge in all_edges:
            if edge.type == "is_a" and edge.weight > 0.8:
                target_node = self.agent.graph.get_node_by_id(edge.target)
                if target_node and len(target_node.name.split()) >= 5:
                    source_node = self.agent.graph.get_node_by_id(edge.source)
                    if source_node:
                        potential_facts.append((source_node, target_node, edge))

        if potential_facts:
            return random.choice(potential_facts)

        return None

    def _deepen_knowledge_of_random_concept(self) -> None:
        """Pick a random known concept and try to learn a new related fact.

        This method serves as the productive fallback for the study cycle
        when no explicit learning goals are present.

        It selects a random, non-trivial concept from the agent's brain,
        searches the web for new information about it, and then feeds any
        newly found factual sentence back into the agent's main `chat`
        method to be learned.
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
        """Find a new, focused, and unknown topic using a heuristic-driven search.

        This method guides the agent's curiosity. It first selects a broad,
        curated subject (e.g., "Physics"), finds related topics on Wikipedia,
        and then applies a series of heuristics to filter for high-quality
        candidates.

        Heuristics include rejecting meta-pages (e.g., "List of...") and
        unpopular/obscure topics by scraping search result counts.

        Args:
            max_attempts: The maximum number of times to try finding a topic.

        Returns:
            A string name of a suitable new topic, or None if none were found.
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
        """Scrape DuckDuckGo to get an approximate search result count for a query.

        This function serves as a heuristic for determining the "popularity"
        or "commonness" of a topic. It scrapes the HTML results page and
        parses the result count.

        Args:
            query: The search term to look up.

        Returns:
            An integer of the approximate number of search results, or None
            if the scrape fails.
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
        """Retrieve a precise definition and part of speech from a dictionary API.

        This is the primary, high-precision tool for resolving "INVESTIGATE"
        goals. It queries a free public dictionary API for a specific word.

        On success, it extracts the primary part of speech and the first
        definition, which is a much more reliable source of linguistic
        knowledge than general web scraping.

        Args:
            word: The single word to define.

        Returns:
            A tuple containing the part of speech and the definition string,
            or None if the word is not found or the API call fails.
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
        """Retrieve and verify a simple fact from a Wikipedia article."""
        print(f"[Knowledge Source]: Searching Wikipedia for '{topic}'...")
        try:
            search_results = wikipedia.search(topic, results=1)
            if not search_results:
                return None

            page = wikipedia.page(search_results[0], auto_suggest=False, redirect=True)
            if not (page and page.summary):
                return None

            if page and page.summary:
                first_sentence = page.summary.split(". ")[0].strip()
                if not first_sentence.endswith("."):
                    first_sentence += "."

            reframed_fact = self.agent.interpreter.verify_and_reframe_fact(
                original_topic=topic,
                raw_sentence=first_sentence,
            )

            if reframed_fact:
                print(
                    f"  [Knowledge Source]: Extracted and verified fact: '{reframed_fact}'",
                )
                return page.title, reframed_fact

        except Exception:
            pass
        return None

    def get_fact_from_duckduckgo(self, topic: str) -> tuple[str, str] | None:
        """Retrieve, verify, and reframe a definition from DuckDuckGo's API."""
        print(f"[Knowledge Source]: Searching DuckDuckGo for '{topic}'...")
        try:
            url = f"https://api.duckduckgo.com/?q={topic}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            definition = data.get("AbstractText") or data.get("Definition")
            if definition:
                first_sentence = definition.split(". ")[0].strip()
                if not first_sentence.endswith("."):
                    first_sentence += "."

                reframed_fact = self.agent.interpreter.verify_and_reframe_fact(
                    original_topic=topic,
                    raw_sentence=first_sentence,
                )

                if reframed_fact:
                    print(
                        f"  [Knowledge Source]: Extracted and verified fact from DuckDuckGo: '{reframed_fact}'",
                    )
                    return topic, reframed_fact

        except Exception:
            pass
        return None
