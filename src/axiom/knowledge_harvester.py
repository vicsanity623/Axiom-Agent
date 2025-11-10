from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import quote

import requests
import wikipedia
from nltk.stem import WordNetLemmatizer

from axiom.dictionary_utils import get_word_info_from_wordnet
from axiom.universal_interpreter import Goal, RelationData

from .knowledge_base import validate_and_add_relation
from .logging_config import console

if TYPE_CHECKING:
    from threading import Lock

    from axiom.cognitive_agent import CognitiveAgent
    from axiom.graph_core import ConceptNode, RelationshipEdge
logger = logging.getLogger(__name__)
wikipedia.set_user_agent("AxiomAgent/1.0 (https://github.com/vicsanity623/Axiom-Agent)")
RESEARCH_CACHE_PATH: Final = Path("data/research_cache.json")


class LogColors:
    GREEN = "\x1b[92m"
    YELLOW = "\x1b[93m"
    WHITE = "\x1b[97m"
    RESET = "\x1b[0m"


class KnowledgeHarvester:
    __slots__ = (
        "agent",
        "lock",
        "lemmatizer",
        "rejected_topics",
        "cache_path",
        "researched_terms",
        "console",
        "llm_definition_failures",
        "llm_circuit_breaker_tripped",
    )

    def __init__(self, agent: CognitiveAgent, lock: Lock) -> None:
        """Initialize the KnowledgeHarvester.

        Args:
            agent: The instance of the CognitiveAgent this harvester will serve.
            lock: A threading lock to ensure thread-safe operations on the agent.
        """
        self.console = console
        self.agent = agent
        self.lock = lock
        self.lemmatizer = WordNetLemmatizer()
        self.rejected_topics: set[str] = set()
        self.cache_path = RESEARCH_CACHE_PATH
        self.researched_terms: set[str] = set()
        self.llm_definition_failures = 0
        self.llm_circuit_breaker_tripped = False
        self._load_research_cache()
        logger.info("[success][Knowledge Harvester]: Initialized.[/success]")

    def _load_research_cache(self) -> None:
        """Load the set of researched terms from a JSON file."""
        if not self.cache_path.exists():
            logger.info(
                "[border][Harvester Cache]: No research cache found. Starting fresh.[/border]"
            )
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.researched_terms = set(data)
                    logger.info(
                        "[border][Harvester Cache]: Loaded %d previously researched terms.[/border]",
                        len(self.researched_terms),
                    )
                else:
                    logger.warning(
                        "[Harvester Cache]: Cache file is malformed; starting fresh."
                    )
        except (OSError, json.JSONDecodeError) as e:
            logger.error(
                "[Harvester Cache]: Failed to load research cache: %s. Starting fresh.",
                e,
            )
            self.researched_terms = set()

    def _save_research_cache(self) -> None:
        """Save the current set of researched terms to a JSON file."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(list(self.researched_terms), f, indent=4)
        except OSError as e:
            logger.error("[Harvester Cache]: Failed to save research cache: %s", e)

    def _mark_as_researched(self, term: str) -> None:
        """Add a term to the research memory and save to disk."""
        if term not in self.researched_terms:
            self.researched_terms.add(term)
            self._save_research_cache()

    def discover_cycle(self) -> None:
        """
        Run one full discovery cycle to find a new topic and create a
        structured "INVESTIGATE" goal.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("\n--- [Discovery Cycle Started at %s] ---", timestamp)
        new_topic = self._find_new_topic()
        if new_topic:
            goal = Goal(task="INVESTIGATE", topic=new_topic)
            with self.lock:
                if goal not in self.agent.learning_goals:
                    self.agent.learning_goals.append(goal)
                    logger.info(
                        "  [Discovery]: Found new topic '%s'. Added to learning goals.",
                        new_topic,
                    )
        else:
            logger.warning(
                "[Discovery Cycle]: Could not find any new topics to learn about this cycle."
            )
        logger.info("--- [Discovery Cycle Finished] ---\n")
        self.agent.log_autonomous_cycle_completion()

    def _resolve_investigation_goal(self, goal: Goal | str) -> bool:
        """
        Attempts to resolve an "INVESTIGATE" goal using a focused, multi-step strategy.
        Returns True on success, False on failure.
        """
        caller_name = f"{self.__class__.__name__}._resolve_investigation_goal"
        term_to_learn: str
        if isinstance(goal, str):
            match = re.match("INVESTIGATE: (.*)", goal)
            if not match:
                return False
            term_to_learn = match.group(1).lower()
        else:
            term_to_learn = goal.get("topic", "").lower()
        if not term_to_learn:
            return False

        if self.agent.lexicon.is_known_word(term_to_learn):
            logger.info(
                "[Study Cycle]: Skipping '%s' — already known.",
                term_to_learn,
            )
            return True

        logger.info(
            "[Study Cycle]: Prioritizing learning goal: To define '%s'.", term_to_learn
        )
        is_single_word = " " not in term_to_learn
        if is_single_word:
            logger.info("   - Term is a single word. Using Dictionary API.")
            api_result = self.get_definition_from_api(term_to_learn)
            if api_result:
                part_of_speech, definition = api_result
                was_learned = False
                with self.lock:
                    pos_relation: RelationData = {
                        "subject": term_to_learn,
                        "verb": "is_a",
                        "object": part_of_speech,
                        "properties": {
                            "provenance": "dictionary_api",
                            "confidence": 0.95,
                        },
                    }
                    pos_learned, _ = self.agent._process_statement_for_learning(
                        pos_relation
                    )
                    if pos_learned:
                        was_learned = True
                    if definition and self.agent.learn_new_fact_autonomously(
                        definition, term_to_learn
                    ):
                        was_learned = True
                if was_learned:
                    logger.info(
                        "[green]   - Successfully learned from Dictionary API.[/green]"
                    )
                    return True
            logger.warning(
                "   - Dictionary API failed for single word '%s'. Marking as unresolvable.",
                term_to_learn,
            )
            self._mark_as_researched(term_to_learn)
            return False
        logger.info("   - Term is a multi-word concept. Using web search.")
        queries = [f"what is {term_to_learn}", f"define {term_to_learn}"]
        for query in queries:
            result = self.get_fact_from_wikipedia(
                query
            ) or self.get_fact_from_duckduckgo(query)
            if result:
                _source, web_fact = result
                with self.lock:
                    if self.agent.learn_new_fact_autonomously(web_fact, term_to_learn):
                        logger.info(
                            "[green]   - Agent successfully learned the new fact from web.[/green]"
                        )
                        return True
                break
            time.sleep(1)
        logger.warning(
            "[yellow]   - All learning attempts failed for '%s'.[/yellow]",
            term_to_learn,
            caller_name,
        )
        self._mark_as_researched(term_to_learn)
        return False

    def _consolidate_pending_knowledge(self, newly_learned_words: list[str]) -> bool:
        """
        Performs a targeted re-evaluation of deferred facts.
        Returns True if all relevant facts were consolidated, False otherwise.
        """
        caller_name = f"{self.__class__.__name__}._consolidate_pending_knowledge"
        if not self.agent.pending_relations or not newly_learned_words:
            return True

        newly_learned_set = set(newly_learned_words)
        relevant_words = newly_learned_set.intersection(
            self.agent.pending_relations.keys()
        )
        if not relevant_words:
            return True

        logger.info(
            "   - Consolidating deferred facts for newly learned words: %s",
            sorted(relevant_words),
        )
        facts_to_reprocess = []
        for word in relevant_words:
            facts_for_word = self.agent.pending_relations.pop(word, [])
            if facts_for_word:
                logger.info(
                    "     - Re-evaluating %d fact(s) blocked by '%s'",
                    len(facts_for_word),
                    word,
                )
                facts_to_reprocess.extend(facts_for_word)

        if not facts_to_reprocess:
            return True

        processed_count = 0
        caller_name = f"{self.__class__.__name__}._consolidate_pending_knowledge"
        for relation, properties in facts_to_reprocess:
            status = validate_and_add_relation(
                agent=self.agent,
                relation=relation,
                properties=properties,
                caller_name=caller_name,
            )
            if status != "deferred":
                processed_count += 1

        if processed_count > 0:
            logger.info(
                "   - Successfully consolidated %d previously deferred facts.",
                processed_count,
            )

        if processed_count < len(facts_to_reprocess):
            logger.warning(
                "   - Consolidation incomplete. %d facts remain deferred.",
                len(facts_to_reprocess) - processed_count,
                caller_name,
            )
            return False

        return True

    def study_cycle(self) -> None:
        """
        Run one full study cycle with a two-tiered priority system and loop detection.
        """
        caller_name = f"{self.__class__.__name__}.study_cycle"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.rule(
            f"[bold cyan]Study Cycle Started at {timestamp}[/bold cyan]", style="cyan"
        )

        task_to_process: Goal | str | None = None
        task_source_is_priority = False
        with self.lock:
            if self.agent.high_priority_tasks:
                task_to_process = self.agent.high_priority_tasks.pop(0)
                task_source_is_priority = True
            elif self.agent.learning_goals:
                task_to_process = self.agent.learning_goals.pop(0)
                task_source_is_priority = False

        if task_to_process:
            task_topic = ""
            task_type = "HIGH-PRIORITY" if task_source_is_priority else "CURRICULUM"
            if isinstance(task_to_process, str):
                task_topic = task_to_process.replace("INVESTIGATE:", "").strip()
            else:
                task_topic = task_to_process.get("topic", "")

            if task_topic:
                logger.info("   - Attempting %s task: '%s'", task_type, task_topic)
                resolved = self._resolve_investigation_goal(task_to_process)

                consolidation_succeeded = True
                if resolved:
                    with self.lock:
                        consolidation_succeeded = self._consolidate_pending_knowledge(
                            [task_topic]
                        )

                if resolved and consolidation_succeeded:
                    logger.info(
                        "   - ✅ %s task '%s' successfully resolved and consolidated.",
                        task_type,
                        task_topic,
                    )
                else:
                    logger.warning(
                        "   - ❌ Failed to fully resolve %s task '%s'. Deprioritizing.",
                        task_type,
                        task_topic,
                        caller_name,
                    )
                    with self.lock:
                        if task_source_is_priority:
                            if isinstance(task_to_process, dict):
                                self.agent.high_priority_tasks.append(task_to_process)
                        else:
                            self.agent.learning_goals.append(task_to_process)
            else:
                logger.warning("Found malformed task: %s. Discarding.", task_to_process)
        else:
            logger.info(
                "All learning queues are empty. Attempting to deepen existing knowledge."
            )
            self._deepen_knowledge_of_random_concept()

        self.console.rule("[bold cyan]Study Cycle Complete.[/bold cyan]", style="cyan")
        self.agent.log_autonomous_cycle_completion()

    def refinement_cycle(self) -> None:
        """
        Run one full introspection and refinement cycle, now including a
        dedicated lexical promotion stage for newly learned concepts.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.rule(
            f"[bold magenta]Refinement Cycle Started at {timestamp}[/bold magenta]",
            style="magenta",
        )

        chunky_fact = None
        with self.lock:
            chunky_fact = self._find_chunky_fact()

        if chunky_fact:
            source_node, target_node, edge = chunky_fact
            logger.info(
                "  [Refinement]: Found a chunky fact to refine: '%s' --[%s]--> '%s'",
                source_node.name,
                edge.type,
                target_node.name,
            )

            full_sentence = f"{source_node.name.capitalize()} {edge.type.replace('_', ' ')} {target_node.name}."

            atomic_sentences = self.agent.interpreter.break_down_definition(
                subject=source_node.name,
                chunky_definition=full_sentence,
            )

            if atomic_sentences:
                logger.info(
                    "  [Refinement]: Decomposed into %d new atomic facts.",
                    len(atomic_sentences),
                )

                facts_learned_count = 0
                newly_learned_concepts = set()

                for sentence in atomic_sentences:
                    with self.lock:
                        if self.agent.learn_new_fact_autonomously(sentence):
                            facts_learned_count += 1
                            for word in sentence.split():
                                clean_word = self.agent._clean_phrase(word)
                                if (
                                    clean_word
                                    and not self.agent.lexicon.is_promoted_word(
                                        clean_word
                                    )
                                ):
                                    newly_learned_concepts.add(clean_word)

                if facts_learned_count > 0:
                    if newly_learned_concepts:
                        logger.info(
                            "  [Refinement]: Promoting %d new concepts to the lexicon...",
                            len(newly_learned_concepts),
                        )
                        for concept in newly_learned_concepts:
                            word_info = get_word_info_from_wordnet(concept)
                            if word_info and isinstance(word_info[0], dict):
                                pos = word_info[0].get("type")
                                if pos in ("noun", "verb", "adjective"):
                                    with self.lock:
                                        self.agent.lexicon.promote_word(
                                            concept, pos, confidence=0.75
                                        )

                    with self.lock:
                        graph_edge_data = self.agent.graph.graph.get_edge_data(
                            edge.source, edge.target
                        )
                        if graph_edge_data:
                            key_to_modify = next(
                                (
                                    key
                                    for key, data in graph_edge_data.items()
                                    if data.get("id") == edge.id
                                ),
                                None,
                            )
                            if key_to_modify is not None:
                                self.agent.graph.graph[edge.source][edge.target][
                                    key_to_modify
                                ]["weight"] = 0.2
                                self.agent.save_brain()
                                logger.info(
                                    "  [Refinement]: Marked original fact as refined by lowering its weight."
                                )
                else:
                    logger.warning(
                        "  [Refinement]: Failed to learn any new atomic facts from the decomposition."
                    )
        else:
            caller_name = f"{self.__class__.__name__}._find_chunky_fact"
            logger.warning(
                "[Refinement]: No chunky facts found for refinement this cycle. (in %s)",
                caller_name,
            )

        self._prune_research_cache()

        self.console.rule(
            "[bold magenta]Refinement Cycle Complete.[/bold magenta]", style="magenta"
        )
        self.agent.log_autonomous_cycle_completion()

    def _prune_research_cache(self) -> None:
        """
        Clean the research cache by removing terms that are now known to the lexicon.
        This is a housekeeping task to prevent the cache from growing indefinitely
        with redundant information.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("\n--- [Refinement Cycle Started at %s] ---", timestamp)
        with self.lock:
            if not self.researched_terms:
                return
            prunable_terms = {
                term
                for term in self.researched_terms
                if self.agent.lexicon.is_known_word(term)
            }
            if prunable_terms:
                logger.info(
                    "  [Cache Pruning]: Removing %d fully learned term(s) from the research cache.",
                    len(prunable_terms),
                )
                self.researched_terms -= prunable_terms
                self._save_research_cache()

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

        def is_chunky(node_name: str) -> bool:
            """A helper to determine if a node's name represents a complex definition."""
            words = node_name.split()
            has_verb_indicator = any(
                word.endswith("s") or word.endswith("ed") or word.endswith("ing")
                for word in words
                if len(word) > 3
            )
            is_long_phrase = len(words) >= 5
            return is_long_phrase or (len(words) >= 3 and has_verb_indicator)

        for edge in all_edges:
            if edge.type in {"is_a", "defines", "describes"} and edge.weight >= 0.8:
                target_node = self.agent.graph.get_node_by_id(edge.target)
                if target_node and is_chunky(target_node.name):
                    source_node = self.agent.graph.get_node_by_id(edge.source)
                    if source_node:
                        potential_facts.append((source_node, target_node, edge))
        if potential_facts:
            return random.choice(potential_facts)
        logger.info(
            "[Refinement]: No chunky facts found with strict criteria, retrying with relaxed threshold..."
        )
        for edge in all_edges:
            if edge.weight >= 0.3:
                target_node = self.agent.graph.get_node_by_id(edge.target)
                if target_node and is_chunky(target_node.name):
                    source_node = self.agent.graph.get_node_by_id(edge.source)
                    if source_node:
                        potential_facts.append((source_node, target_node, edge))
        return random.choice(potential_facts) if potential_facts else None

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
                logger.info(
                    "[Deepen Knowledge]: Not enough concepts in the brain to study yet."
                )
                return
            _, node_data = random.choice(all_nodes)
            random_node_name = node_data.get("name")
        if not random_node_name or random_node_name in stop_words:
            if random_node_name:
                logger.info(
                    "[Deepen Knowledge]: Skipping study of common concept: '%s'",
                    random_node_name,
                )
            return
        logger.info(
            "[Deepen Knowledge]: Chosen to study the concept: '%s'", random_node_name
        )
        result = self.get_fact_from_wikipedia(
            random_node_name
        ) or self.get_fact_from_duckduckgo(random_node_name)
        if result:
            _title, fact_sentence = result
            with self.lock:
                logger.info(
                    "  [Deepen Knowledge]: %sAttempting to learn new fact: '%s'%s",
                    LogColors.GREEN,
                    fact_sentence,
                    LogColors.RESET,
                )
                self.agent.chat(fact_sentence)
        else:
            logger.warning(
                "[Deepen Knowledge]: Could not find any new facts about '%s'.",
                random_node_name,
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
            logger.info(
                "[Discovery]: Searching for a new topic (Attempt %d/%d)...",
                i + 1,
                max_attempts,
            )
            try:
                subject = random.choice(core_subjects)
                logger.info("  [Discovery]: Exploring the core subject: '%s'", subject)
                related_topics = wikipedia.search(subject, results=10)
                if not related_topics:
                    continue
                topic = random.choice(related_topics)
                reject_keywords = ["list of", "timeline of", "index of", "outline of"]
                if any(keyword in topic.lower() for keyword in reject_keywords):
                    logger.info(
                        "  [Discovery Heuristic]: Rejecting meta-page topic: '%s'",
                        topic,
                    )
                    continue
                search_popularity = self._get_search_result_count(topic)
                minimum_popularity = 10000
                if (
                    search_popularity is not None
                    and search_popularity < minimum_popularity
                ):
                    logger.info(
                        "  [Discovery Heuristic]: Rejecting obscure topic '%s' (popularity: %d)",
                        topic,
                        search_popularity,
                    )
                    continue
                clean_topic = self.agent._clean_phrase(topic)
                if (
                    not self.agent.lexicon.is_known_word(clean_topic)
                    and clean_topic not in self.rejected_topics
                ):
                    logger.info(
                        "  [Discovery Success]: Found new, popular topic: '%s'", topic
                    )
                    return clean_topic
            except Exception as e:
                logger.warning(
                    "  [Discovery Error]: An error occurred during topic finding. Error: %s",
                    e,
                )
                time.sleep(1)
        logger.warning(
            "[Discovery Warning]: Could not find a new, suitable topic after %d attempts.",
            max_attempts,
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
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            }
            url = f"https://duckduckgo.com/html/?q={quote(query)}"
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            match = re.search("([0-9,]+) results", response.text)
            if match:
                count_str = match.group(1).replace(",", "")
                return int(count_str)
        except Exception:
            return None
        return None

    def get_definition_from_api(
        self, word: str, context: str | None = None
    ) -> tuple[str, str] | None:
        """
        Retrieve a precise definition and part of speech from the reliable
        dictionaryapi.dev. The LLM-based context-aware feature has been
        disabled due to unreliability.
        """
        logger.info("[yellow]   - Querying for definition of[/yellow] '%s'...", word)
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                logger.warning("     - Word not found in Dictionary API.")
                return None
            data = response.json()
            if data and isinstance(data, list):
                word_entry = data[0]
                if "meanings" in word_entry and word_entry["meanings"]:
                    first_meaning = word_entry["meanings"][0]
                    part_of_speech = first_meaning.get("partOfSpeech")
                    if "definitions" in first_meaning and first_meaning["definitions"]:
                        definition = first_meaning["definitions"][0].get("definition")
                        if part_of_speech and definition:
                            logger.info(
                                "   - ✅ Found definition via Dictionary API: '%s'",
                                definition,
                            )
                            return (part_of_speech, definition)
        except requests.RequestException as e:
            logger.error(
                "[error][Dictionary API]: A network error occurred: %s[/error]", e
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(
                "[error][Dictionary API]: Failed to parse response for '%s': %s[/error]",
                word,
                e,
            )
        logger.warning(
            "     - All attempts to find a definition for '%s' failed.", word
        )
        return None

    def get_fact_from_wikipedia(self, topic: str) -> tuple[str, str] | None:
        """Retrieve and verify a simple fact from a Wikipedia article."""
        logger.info("[Knowledge Source]: Searching Wikipedia for '{topic}'...")
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
            if self.agent.interpreter.llm is not None:
                reframed_fact = self.agent.interpreter.verify_and_reframe_fact(
                    original_topic=topic, raw_sentence=first_sentence
                )
            else:
                logger.warning(
                    "  [Fact Verifier Error]: LLM is disabled. Cannot verify/reframe. Skipping reframing."
                )
                reframed_fact = first_sentence
            if reframed_fact:
                logger.info(
                    "  [Knowledge Source]: Extracted and verified fact: '{reframed_fact}'"
                )
                return (page.title, reframed_fact)
        except Exception:
            pass
        return None

    def get_fact_from_duckduckgo(self, topic: str) -> tuple[str, str] | None:
        """Retrieve, verify, and reframe a definition from DuckDuckGo's API."""
        logger.info("[Knowledge Source]: Searching DuckDuckGo for '%s'...", topic)
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
                    original_topic=topic, raw_sentence=first_sentence
                )
                if reframed_fact:
                    logger.info(
                        "  [Knowledge Source]: Extracted and verified fact from DuckDuckGo: '%s'",
                        reframed_fact,
                    )
                    return (topic, reframed_fact)
        except Exception as e:
            logger.debug(
                "  [Knowledge Source]: DuckDuckGo search failed for '%s'. Error: %s",
                topic,
                e,
            )
            pass
        return None
