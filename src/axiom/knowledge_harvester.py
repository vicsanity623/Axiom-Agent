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

from axiom.universal_interpreter import PropertyData, RelationData, Task

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
        "_failed_tasks",
    )

    def __init__(self, agent: CognitiveAgent, lock: Lock) -> None:
        """Initialize the KnowledgeHarvester."""
        self.console = console
        self.agent = agent
        self.lock = lock
        self.lemmatizer = WordNetLemmatizer()
        self.rejected_topics: set[str] = set()
        self.cache_path = RESEARCH_CACHE_PATH
        self.researched_terms: set[str] = set()
        self.llm_definition_failures = 0
        self.llm_circuit_breaker_tripped = False
        self._failed_tasks: set[str] = set()
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
            task = Task(task="INVESTIGATE", topic=new_topic)
            with self.lock:
                if task not in self.agent.learning_goals:
                    self.agent.learning_goals.append(task)
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

    def _resolve_investigation_goal(self, goal: Task | str) -> bool:
        """
        Attempts to resolve an "INVESTIGATE" goal using a focused, multi-step strategy
        with a robust fallback from Dictionary API to web search.
        Returns True on success, False on failure.
        """
        caller_name = f"{self.__class__.__name__}._resolve_investigation_goal"
        raw_term: str
        if isinstance(goal, str):
            match = re.match("INVESTIGATE: (.*)", goal)
            if not match:
                logger.warning(f"Malformed string goal received: {goal}. Discarding.")
                return False
            raw_term = match.group(1)
        else:
            raw_term = goal.get("topic", "")

        term_to_learn = self.agent._clean_phrase(raw_term)

        if not term_to_learn:
            logger.warning(
                f"Received goal with no valid topic after cleaning: {goal}. Discarding."
            )
            return False

        logger.info(
            f"[Study Cycle]: Prioritizing learning goal: To define '{term_to_learn}' (from '{raw_term}')."
        )

        is_single_word = " " not in term_to_learn
        if is_single_word:
            logger.info(
                f"   - Term is a single word. Attempting Dictionary API first for '{term_to_learn}'."
            )
            api_result = self.get_definition_from_api(term_to_learn)
            if api_result:
                part_of_speech, definition = api_result
                was_learned = False
                with self.lock:
                    self.agent.lexicon.promote_word(
                        term_to_learn, part_of_speech, confidence=0.9, quiet=True
                    )

                    pos_relation: RelationData = {
                        "subject": term_to_learn,
                        "verb": "is_a",
                        "object": part_of_speech,
                    }
                    pos_properties: PropertyData = {
                        "provenance": "dictionary_api",
                        "confidence": 0.95,
                    }
                    status = validate_and_add_relation(
                        self.agent, pos_relation, pos_properties, caller_name
                    )
                    if status != "deferred":
                        was_learned = True

                    if definition and self.agent.learn_new_fact_autonomously(
                        definition, term_to_learn
                    ):
                        was_learned = True

                if was_learned:
                    logger.info(
                        f"[green]   - Successfully learned '{term_to_learn}' from Dictionary API.[/green]"
                    )
                    return True
            else:
                logger.warning(
                    f"   - Dictionary API failed for '{term_to_learn}'. Falling back to web search."
                )

        logger.info(f"   - Using web search for '{term_to_learn}'.")
        queries = [f"what is {term_to_learn}", f"define {term_to_learn}"]
        for query in queries:
            result = self.get_fact_from_wikipedia(
                query
            ) or self.get_fact_from_duckduckgo(query)
            if result:
                _source, web_fact = result
                with self.lock:
                    if self.agent.learn_new_fact_autonomously(web_fact, term_to_learn):
                        self.agent.lexicon.promote_word(
                            term_to_learn, "noun", confidence=0.7, quiet=True
                        )

                        logger.info(
                            f"[green]   - Agent successfully learned '{term_to_learn}' from web.[/green]"
                        )
                        return True
                break
            time.sleep(1)

        logger.warning(
            f"[yellow]   - All learning attempts failed for '{term_to_learn}'. (in {caller_name})[/yellow]"
        )
        self._mark_as_researched(term_to_learn)
        return False

    def _consolidate_pending_knowledge(self, newly_learned_words: list[str]) -> bool:
        """
        Performs a targeted re-evaluation of deferred facts.
        Returns True if any progress was made, False otherwise.
        """
        if not self.agent.pending_relations or not newly_learned_words:
            return False
        newly_learned_set = set(newly_learned_words)
        relevant_words = newly_learned_set.intersection(
            self.agent.pending_relations.keys()
        )
        if not relevant_words:
            return False
        logger.info(
            f"   - Consolidating deferred facts for newly learned words: {sorted(relevant_words)}"
        )
        facts_to_reprocess = []
        for word in relevant_words:
            facts_for_word = self.agent.pending_relations.pop(word, [])
            if facts_for_word:
                logger.info(
                    f"     - Re-evaluating {len(facts_for_word)} fact(s) blocked by '{word}'"
                )
                facts_to_reprocess.extend(facts_for_word)
        if not facts_to_reprocess:
            return False
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
                f"   - Successfully consolidated {processed_count} previously deferred facts."
            )
        return processed_count > 0

    def study_cycle(self, phase_end_time: datetime | None = None) -> bool:
        """
        Run one full, recursive study cycle until all possible knowledge is consolidated,
        while respecting an optional phase end time.

        Returns:
            bool: True if progress was made (task processed, knowledge consolidated),
                False if the cycle was stalled or no work was done.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.rule(
            f"[bold cyan]Study Cycle Started at {timestamp}[/bold cyan]", style="cyan"
        )
        progress_made_in_cycle = False
        cycle_iteration = 0

        while True:
            if phase_end_time and datetime.now() >= phase_end_time:
                logger.warning(
                    "   - Study cycle time limit reached. Yielding control to CycleManager."
                )
                break

            cycle_iteration += 1
            logger.info(f"--- Starting Study Cycle Iteration #{cycle_iteration} ---")

            task_to_process: Task | str | None = None
            task_type = ""
            with self.lock:
                if self.agent.high_priority_tasks:
                    task_to_process = self.agent.high_priority_tasks.pop(0)
                    task_type = "HIGH-PRIORITY"
                elif self.agent.learning_goals:
                    task_to_process = self.agent.learning_goals.pop(0)
                    task_type = "CURRICULUM"

            if task_to_process:
                task_topic = (
                    task_to_process.replace("INVESTIGATE:", "").strip()
                    if isinstance(task_to_process, str)
                    else task_to_process.get("topic", "")
                )

                if not task_topic:
                    logger.warning(
                        f"Found malformed task: {task_to_process}. Discarding."
                    )
                else:
                    logger.info(f"   - Attempting {task_type} task: '{task_topic}'")
                    resolved = self._resolve_investigation_goal(task_to_process)
                    if not resolved:
                        logger.warning(
                            f"   - ❌ Task '{task_topic}' failed at resolution stage. Deprioritizing."
                        )
                        self._failed_tasks.add(task_topic)
                        with self.lock:
                            self.agent.learning_goals.append(task_to_process)
                    else:
                        logger.info(
                            f"   - ✅ Task '{task_topic}' resolved. Knowledge is pending consolidation."
                        )
                        progress_made_in_cycle = True
            else:
                logger.info(
                    "All learning queues are empty. Attempting to deepen existing knowledge."
                )
                if self._deepen_knowledge_of_random_concept():
                    progress_made_in_cycle = True

            logger.info("   - Initiating iterative knowledge consolidation phase...")
            consolidation_made_progress = False
            with self.lock:
                while self.agent.pending_relations:
                    num_pending_before = len(self.agent.pending_relations)
                    all_known_concepts = self.agent.graph.get_all_node_names()
                    self._consolidate_pending_knowledge(all_known_concepts)
                    num_pending_after = len(self.agent.pending_relations)

                    if num_pending_after >= num_pending_before:
                        break
                    consolidation_made_progress = True
                    progress_made_in_cycle = True

            if self.agent.pending_relations:
                logger.warning(
                    f"   - Consolidation stalled. {len(self.agent.pending_relations)} concepts still have deferred facts."
                )
                blocker_word_raw = next(iter(self.agent.pending_relations.keys()), None)
                if blocker_word_raw:
                    blocker_word_clean = self.agent._clean_phrase(blocker_word_raw)
                    if not blocker_word_clean:
                        logger.error(
                            f"   - Could not clean blocker word '{blocker_word_raw}'. Exiting cycle."
                        )
                        break

                    if blocker_word_clean in self._failed_tasks:
                        logger.error(
                            f"   - Stalled on '{blocker_word_clean}', which has failed recently. "
                            "Breaking cycle to prevent loop."
                        )
                        break

                    logger.info(
                        f"   - Autonomously creating new HIGH-PRIORITY task to unblock consolidation: 'INVESTIGATE: {blocker_word_clean}'"
                    )
                    new_task: Task = {
                        "task": "INVESTIGATE",
                        "topic": blocker_word_clean,
                    }
                    with self.lock:
                        if new_task not in self.agent.high_priority_tasks:
                            self.agent.high_priority_tasks.insert(0, new_task)
                    continue
                logger.error(
                    "   - Consolidation stalled but could not identify a blocker. Exiting cycle."
                )
                break

            if not self.agent.pending_relations and consolidation_made_progress:
                logger.info("   - ✅ All pending knowledge successfully consolidated.")
                if self._failed_tasks:
                    logger.info("      - Clearing recent task failure memory.")
                    self._failed_tasks.clear()

            logger.info("   - Study cycle iteration complete.")
            break

        self.console.rule("[bold cyan]Study Cycle Complete.[/bold cyan]", style="cyan")
        self.agent.log_autonomous_cycle_completion()
        return progress_made_in_cycle

    def refinement_cycle(self) -> None:
        """
        Run one full introspection and refinement cycle.
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
                subject=source_node.name, chunky_definition=full_sentence
            )
            if atomic_sentences:
                logger.info(
                    "  [Refinement]: Decomposed into %d new atomic facts.",
                    len(atomic_sentences),
                )
                facts_learned_count = 0
                for sentence in atomic_sentences:
                    with self.lock:
                        if self.agent.learn_new_fact_autonomously(sentence):
                            facts_learned_count += 1
                if facts_learned_count > 0:
                    with self.lock:
                        self.agent.graph.update_edge_properties(
                            edge, {"revision_status": "superseded"}
                        )
                        self.agent.save_brain()
                        logger.info(
                            "  [Refinement]: Marked original fact as refined by superseding it."
                        )
                else:
                    logger.warning(
                        "  [Refinement]: Failed to learn any new atomic facts from the decomposition."
                    )
        else:
            caller_name = f"{self.__class__.__name__}._find_chunky_fact"
            logger.warning(
                f"[Refinement]: No chunky facts found for refinement this cycle. (in {caller_name})"
            )
        self._prune_research_cache()
        self.console.rule(
            "[bold magenta]Refinement Cycle Complete.[/bold magenta]", style="magenta"
        )
        self.agent.log_autonomous_cycle_completion()

    def _prune_research_cache(self) -> None:
        """
        Clean the research cache by removing terms that are now known to the lexicon.
        """
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
        """Find a single, high-confidence, unrefined "chunky" fact."""
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
            if edge.properties.get("revision_status") == "superseded":
                continue

            if edge.type in {"is_a", "defines", "describes"} and edge.weight >= 0.8:
                target_node = self.agent.graph.get_node_by_id(edge.target)
                if target_node and is_chunky(target_node.name):
                    source_node = self.agent.graph.get_node_by_id(edge.source)
                    if source_node:
                        potential_facts.append((source_node, target_node, edge))

        if potential_facts:
            return random.choice(potential_facts)

        return None

    def _deepen_knowledge_of_random_concept(self) -> bool:
        """
        Pick a random known concept and try to learn a new related fact.

        Returns:
            bool: True if a new fact was successfully found and queued for learning,
                False otherwise.
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
        random_node_name: str | None = None
        with self.lock:
            all_node_names = self.agent.graph.get_all_node_names()
            if len(all_node_names) < 2:
                logger.info(
                    "[Deepen Knowledge]: Not enough concepts in the brain to study yet."
                )
                return False

            eligible_names = [
                name
                for name in all_node_names
                if name not in stop_words and " " not in name
            ]
            if not eligible_names:
                logger.info(
                    "[Deepen Knowledge]: No eligible single-word concepts to study."
                )
                return False
            random_node_name = random.choice(eligible_names)

        if not random_node_name:
            return False

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
                    f"  [Deepen Knowledge]: {LogColors.GREEN}Attempting to learn new fact: '{fact_sentence}'{LogColors.RESET}"
                )
                self.agent.chat(fact_sentence)
            return True
        logger.warning(
            "[Deepen Knowledge]: Could not find any new facts about '%s'.",
            random_node_name,
        )
        return False

    def _find_new_topic(self, max_attempts: int = 5) -> str | None:
        """Find a new, focused, and unknown topic using a heuristic-driven search."""
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
        """Scrape DuckDuckGo to get an approximate search result count for a query."""
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
        Retrieve a precise definition and part of speech from dictionaryapi.dev.
        """
        logger.info(f"[yellow]   - Querying for definition of[/yellow] '{word}'...")
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
                f"[error][Dictionary API]: A network error occurred: {e}[/error]"
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(
                f"[error][Dictionary API]: Failed to parse response for '{word}': {e}[/error]"
            )
        logger.warning(f"     - All attempts to find a definition for '{word}' failed.")
        return None

    def get_fact_from_wikipedia(self, topic: str) -> tuple[str, str] | None:
        """Retrieve and verify a simple fact from a Wikipedia article."""
        logger.info(f"[Knowledge Source]: Searching Wikipedia for '{topic}'...")
        try:
            search_results = wikipedia.search(topic, results=1)
            if not search_results:
                return None
            page = wikipedia.page(search_results[0], auto_suggest=False, redirect=True)
            if not (page and page.summary):
                return None
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
                    f"  [Knowledge Source]: Extracted and verified fact: '{reframed_fact}'"
                )
                return (page.title, reframed_fact)
        except Exception:
            pass
        return None

    def get_fact_from_duckduckgo(self, topic: str) -> tuple[str, str] | None:
        """Retrieve, verify, and reframe a definition from DuckDuckGo's API."""
        logger.info(f"[Knowledge Source]: Searching DuckDuckGo for '{topic}'...")
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
