from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import quote

import requests
import wikipedia
from nltk.stem import WordNetLemmatizer

from axiom.universal_interpreter import PropertyData, Task

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
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(list(self.researched_terms), f, indent=4)
        except OSError as e:
            logger.error("[Harvester Cache]: Failed to save research cache: %s", e)

    def _mark_as_researched(self, term: str) -> None:
        if term not in self.researched_terms:
            self.researched_terms.add(term)
            self._save_research_cache()

    def discover_cycle(self) -> None:
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
            api_result = self.get_definition_from_api(term_to_learn)
            if api_result:
                part_of_speech, definition = api_result
                was_learned = False
                with self.lock:
                    self.agent.lexicon.add_linguistic_knowledge(
                        term_to_learn,
                        part_of_speech,
                        confidence=0.9,
                        provenance="dictionary_api",
                    )
                    sub_node = self.agent.knowledge_manager.get_or_create_concept(
                        term_to_learn
                    )
                    obj_node = self.agent.knowledge_manager.get_or_create_concept(
                        part_of_speech
                    )
                    if sub_node and obj_node:
                        pos_properties: PropertyData = {
                            "provenance": "dictionary_api",
                            "confidence": 0.95,
                        }
                        status = validate_and_add_relation(
                            agent=self.agent,
                            sub_node=sub_node,
                            obj_node=obj_node,
                            relation_type="is a",
                            properties=pos_properties,
                            caller_name=caller_name,
                        )
                        if status != "deferred":
                            was_learned = True
                    if (
                        definition
                        and self.agent.knowledge_manager.learn_new_fact_autonomously(
                            definition, term_to_learn
                        )
                    ):
                        was_learned = True
                if was_learned:
                    logger.info(
                        f"[green]   - Successfully learned '{term_to_learn}' from Dictionary API.[/green]"
                    )
                    return True  # <-- This return is now correctly hit
            else:
                logger.warning(
                    f"   - Dictionary API failed for '{term_to_learn}'. Falling back to web search."
                )

        # --- FIX: Separate the search loop from the learning logic ---

        # Phase 1: Search for a valid web fact
        web_fact: str | None = None
        queries = [f"what is {term_to_learn}", f"define {term_to_learn}"]
        for query in queries:
            result = self.get_fact_from_wikipedia(
                query
            ) or self.get_fact_from_duckduckgo(query)
            if result:
                _source, web_fact = result
                break  # Found a result, stop searching
            time.sleep(1)

        # Phase 2: Attempt to learn from the fact, if one was found
        if web_fact:
            with self.lock:
                if self.agent.knowledge_manager.learn_new_fact_autonomously(
                    web_fact, term_to_learn
                ):
                    self.agent.lexicon.add_linguistic_knowledge(
                        term_to_learn, "noun", confidence=0.7, provenance="web_search"
                    )
                    logger.info(
                        f"[green]   - Agent successfully learned '{term_to_learn}' from web.[/green]"
                    )
                    return True

        logger.warning(
            f"[yellow]   - All learning attempts failed for '{term_to_learn}'. (in {caller_name})[/yellow]"
        )
        self._mark_as_researched(term_to_learn)
        return False

    def _consolidate_pending_knowledge(self) -> bool:
        """
        Performs a full re-evaluation of ALL deferred facts against the
        agent's complete, current lexicon.
        Returns True if any progress was made, False otherwise.
        """
        if not self.agent.pending_relations:
            return False
        all_known_words = self.agent.graph.get_all_node_names()
        facts_to_reprocess = []
        words_to_clear = []
        for blocker_word, facts in self.agent.pending_relations.items():
            if blocker_word in all_known_words:
                facts_to_reprocess.extend(facts)
                words_to_clear.append(blocker_word)
        if not facts_to_reprocess:
            return False
        logger.info(
            f"   - Consolidating deferred facts unblocked by: {sorted(words_to_clear)}"
        )
        for word in words_to_clear:
            self.agent.pending_relations.pop(word, None)
        processed_count = 0
        caller_name = f"{self.__class__.__name__}._consolidate_pending_knowledge"
        unique_fact_strings = set()
        for relation, properties in facts_to_reprocess:
            fact_string = json.dumps((relation, properties), sort_keys=True)
            unique_fact_strings.add(fact_string)
        if not unique_fact_strings:
            return False
        for fact_string in unique_fact_strings:
            relation, properties = json.loads(fact_string)
            subject_name = self.agent._extract_name_from_relation(
                relation.get("subject")
            )
            object_name = self.agent._extract_name_from_relation(relation.get("object"))
            relation_type = self.agent._extract_name_from_relation(relation.get("verb"))
            if not subject_name or not object_name or (not relation_type):
                logger.warning(f"   - Skipping malformed deferred fact: {relation}")
                continue
            sub_node = self.agent.knowledge_manager.get_or_create_concept(subject_name)
            obj_node = self.agent.knowledge_manager.get_or_create_concept(object_name)
            if not sub_node or not obj_node:
                logger.warning(
                    f"   - Could not create nodes for deferred fact: {relation}"
                )
                continue
            status = validate_and_add_relation(
                agent=self.agent,
                sub_node=sub_node,
                obj_node=obj_node,
                relation_type=relation_type,
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
        Run a full, priority-driven study cycle until no further progress can be made.

        This cycle continuously makes the most efficient choice based on the agent's
        current state, prioritizing actions in the following order:

        1.  **Iterative Consolidation**: The highest priority is always to integrate
            pending facts. The agent will remain in a tight consolidation loop until
            no more facts can be integrated in a pass.

        2.  **Task Processing**: If no consolidation is possible, the agent processes
            a single task from its high-priority or curriculum queues. Any new
            knowledge gained immediately triggers a return to the consolidation phase.

        3.  **Stall Resolution**: If the agent is idle (no tasks to process) but
            consolidation is still stalled, it creates a batch of tasks for ALL
            unblocking concepts to ensure it can make progress in the next iteration.

        4.  **Proactive Deepening**: If truly idle (no tasks, no pending facts), the
            agent expands on existing concepts to discover new information.

        The cycle terminates if a full pass occurs with no progress, or if a hard
        stall on unresolvable blockers is detected.

        Returns:
            bool: True if any knowledge was gained or consolidated during the cycle.
        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.rule(
            f"[bold cyan]Study Cycle Started at {timestamp}[/bold cyan]", style="cyan"
        )
        progress_made_in_cycle = False
        cycle_iteration = 0

        while True:
            cycle_iteration += 1
            logger.info(f"--- Starting Study Cycle Iteration #{cycle_iteration} ---")

            if phase_end_time and datetime.now() >= phase_end_time:
                logger.warning("   - Study cycle time limit reached. Yielding control.")
                break

            # PRIORITY 1: Always try to consolidate first.
            if self.agent.pending_relations:
                logger.info("   - Priority 1: Attempting knowledge consolidation.")
                num_pending_at_start = len(self.agent.pending_relations)

                with self.lock:
                    while self.agent.pending_relations:
                        if not self._consolidate_pending_knowledge():
                            break

                consolidated_count = num_pending_at_start - len(
                    self.agent.pending_relations
                )
                if consolidated_count > 0:
                    logger.info(
                        f"   - ✅ Consolidation successful. {consolidated_count} facts integrated."
                    )
                    progress_made_in_cycle = True
                    continue

            # PRIORITY 2: If consolidation is stalled, process a task to unblock it.
            raw_task: Task | str | None = None
            task_type = ""
            with self.lock:
                if self.agent.high_priority_tasks:
                    raw_task = self.agent.high_priority_tasks.pop(0)
                    task_type = "HIGH-PRIORITY"
                elif self.agent.learning_goals:
                    raw_task = self.agent.learning_goals.pop(0)
                    task_type = "CURRICULUM"

            if raw_task:
                task_to_process: Task | None = None
                if isinstance(raw_task, str):
                    topic = raw_task.replace("INVESTIGATE:", "").strip()
                    if topic:
                        task_to_process = cast(
                            "Task", {"task": "INVESTIGATE", "topic": topic}
                        )
                elif isinstance(raw_task, dict):
                    task_to_process = raw_task

                if task_to_process:
                    task_topic = task_to_process.get("topic", "N/A")
                    logger.info(
                        f"   - Priority 2: Attempting {task_type} task: '{task_topic}'"
                    )
                    if self._resolve_investigation_goal(task_to_process):
                        logger.info(
                            f"   - ✅ Task '{task_topic}' resolved. Knowledge pending."
                        )
                        progress_made_in_cycle = True
                        continue
                    logger.warning(
                        f"   - ❌ Task '{task_topic}' failed. Deprioritizing."
                    )
                    self._failed_tasks.add(task_topic)
                    with self.lock:
                        self.agent.learning_goals.append(task_to_process)
                    progress_made_in_cycle = True
                    continue

            # PRIORITY 3: If idle but still stalled, generate unblocking tasks.
            if self.agent.pending_relations:
                logger.warning(
                    "   - Priority 3: Idle but stalled. Generating unblocking tasks."
                )
                unblocking_tasks_created = False
                new_tasks_to_add = []
                topics_already_queued = {
                    task.get("topic")
                    for task in self.agent.high_priority_tasks
                    if isinstance(task, dict)
                }
                topics_in_this_batch = set()

                for blocker_raw in list(self.agent.pending_relations.keys()):
                    blocker_clean = self.agent._clean_phrase(blocker_raw)
                    if blocker_clean and blocker_clean not in self._failed_tasks:
                        if (
                            blocker_clean not in topics_already_queued
                            and blocker_clean not in topics_in_this_batch
                        ):
                            new_task = cast(
                                "Task", {"task": "INVESTIGATE", "topic": blocker_clean}
                            )
                            logger.info(
                                f"   - Autonomously creating HIGH-PRIORITY task to unblock: '{blocker_clean}'"
                            )
                            new_tasks_to_add.append(new_task)
                            topics_in_this_batch.add(blocker_clean)
                            unblocking_tasks_created = True

                if new_tasks_to_add:
                    with self.lock:
                        self.agent.high_priority_tasks.extend(new_tasks_to_add)

                if unblocking_tasks_created:
                    progress_made_in_cycle = True
                    continue

                logger.error(
                    "   - Hard Stall: Unable to resolve blockers. All have failed recently. Breaking cycle."
                )
                break

            # PRIORITY 4: If truly idle, attempt proactive deepening.
            if self._deepen_knowledge_of_random_concept():
                progress_made_in_cycle = True
                continue

            logger.info("   - No progress made in iteration. Study cycle is complete.")
            break

        self.console.rule("[bold cyan]Study Cycle Complete.[/bold cyan]", style="cyan")
        self.agent.log_autonomous_cycle_completion()
        return progress_made_in_cycle

    def refinement_cycle(self) -> None:
        """
        Periodically reviews and improves the quality of the knowledge graph by
        decomposing complex facts.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.rule(
            f"[bold magenta]Refinement Cycle Started at {timestamp}[/bold magenta]",
            style="magenta",
        )
        actions_taken = 0
        chunky_fact = None
        with self.lock:
            chunky_fact = self._find_chunky_fact()

        if chunky_fact:
            actions_taken += 1
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
                facts_learned_count = 0
                for sentence in atomic_sentences:
                    with self.lock:
                        # --- FIX: Correctly handle the single boolean return value ---
                        if self.agent.knowledge_manager.learn_new_fact_autonomously(
                            sentence
                        ):
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
            logger.info(
                "[Refinement]: No chunky facts found for refinement this cycle."
            )

        self._prune_research_cache()

        if actions_taken == 0:
            logger.info("   - No refinement actions were necessary in this cycle.")

        self.console.rule(
            "[bold magenta]Refinement Cycle Complete.[/bold magenta]", style="magenta"
        )
        self.agent.log_autonomous_cycle_completion()

    def _prune_research_cache(self) -> None:
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
        potential_facts = []
        all_edges = self.agent.graph.get_all_edges()

        def is_chunky(node_name: str) -> bool:
            words = node_name.split()
            has_verb_indicator = any(
                word.endswith(("s", "ed", "ing")) for word in words if len(word) > 3
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

    def _deepen_knowledge_of_random_concept(self, max_attempts: int = 5) -> bool:
        """
        Attempts to find and learn a new fact about a randomly chosen existing
        concept. Tries multiple times if a chosen concept yields no new info.
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

        with self.lock:
            all_node_names = self.agent.graph.get_all_node_names()
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

        for attempt in range(max_attempts):
            if not eligible_names:
                break

            random_node_name = random.choice(eligible_names)
            logger.info(
                "[Deepen Knowledge]: Attempt %d/%d: Chosen to study the concept: '%s'",
                attempt + 1,
                max_attempts,
                random_node_name,
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
                    # Use the chat method as per your original logic
                    self.agent.conversation_manager.chat(fact_sentence)
                # If we successfully find and process a fact, our job is done.
                return True

            eligible_names.remove(random_node_name)

        logger.warning(
            "[Deepen Knowledge]: Could not find any new facts after %d attempts.",
            max_attempts,
        )
        return False

    def _find_new_topic(self, max_attempts: int = 5) -> str | None:
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
        Queries a dictionary API for a word's definition, trying multiple
        variations (as-is, capitalized, lemmatized) to maximize success rate.
        """
        logger.info(f"[yellow]   - Querying for definition of[/yellow] '{word}'...")
        candidates = [word, word.capitalize()]
        lemma = self.agent.lemmatizer.lemmatize(word, pos="n")
        if lemma != word:
            candidates.append(lemma)
        unique_candidates = list(dict.fromkeys(candidates))
        for candidate in unique_candidates:
            try:
                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{candidate}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list):
                        word_entry = data[0]
                        if "meanings" in word_entry and word_entry["meanings"]:
                            first_meaning = word_entry["meanings"][0]
                            part_of_speech = first_meaning.get("partOfSpeech")
                            if (
                                "definitions" in first_meaning
                                and first_meaning["definitions"]
                            ):
                                definition = first_meaning["definitions"][0].get(
                                    "definition"
                                )
                                if part_of_speech and definition:
                                    logger.info(
                                        "   - ✅ Found definition via Dictionary API for '%s': '%s'",
                                        candidate,
                                        definition,
                                    )
                                    return (part_of_speech, definition)
            except requests.RequestException as e:
                logger.error(
                    f"[error][Dictionary API]: A network error occurred: {e}[/error]"
                )
                return None
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(
                    f"[error][Dictionary API]: Failed to parse response for '{candidate}': {e}[/error]"
                )
                continue
        logger.warning(
            f"     - Word not found in Dictionary API after trying: {unique_candidates}"
        )
        return None

    def get_fact_from_wikipedia(self, topic: str) -> tuple[str, str] | None:
        caller_name = f"{self.__class__.__name__}.get_fact_from_wikipedia"
        logger.info(f"[Knowledge Source]: Searching Wikipedia for '{topic}'...")
        try:
            search_results = wikipedia.search(topic, results=1)
            if not search_results:
                logger.warning(f"   - No Wikipedia results found for '{topic}'.")
                return None

            page = wikipedia.page(search_results[0], auto_suggest=False, redirect=True)

            if not (page and page.summary):
                logger.warning(f"   - Wikipedia page for '{topic}' has no summary.")
                return None

            first_sentence = page.summary.split(". ")[0].strip()
            if not first_sentence.endswith("."):
                first_sentence += "."

            if self.agent.interpreter.llm is not None:
                reframed_fact = self.agent.interpreter.verify_and_reframe_fact(
                    original_topic=topic,
                    raw_sentence=first_sentence,
                )
            else:
                reframed_fact = first_sentence

            if reframed_fact:
                logger.info(
                    f"  [Knowledge Source]: Extracted and verified fact: '{reframed_fact}'"
                )
                return (page.title, reframed_fact)

        except wikipedia.exceptions.PageError:
            logger.warning(f"   - Wikipedia page not found for '{topic}'.")
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(
                f"   - Wikipedia search for '{topic}' was ambiguous: {e.options[:3]}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"   - [Wikipedia API]: A network error occurred: {e}")
        except Exception as e:
            logger.error(
                f"   - [Wikipedia API]: An unexpected error occurred for topic '{topic}': {e}",
                caller_name,
                exc_info=True,
            )

        return None

    def get_fact_from_duckduckgo(self, topic: str) -> tuple[str, str] | None:
        logger.info(f"[Knowledge Source]: Searching DuckDuckGo for '{topic}'...")
        try:
            url = f"https://api.duckduckgo.com/?q={quote(topic)}&format=json&no_html=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            definition = data.get("AbstractText") or data.get("Definition")
            if definition:
                first_sentence = definition.split(". ")[0].strip()
                if not first_sentence.endswith("."):
                    first_sentence += "."

                if self.agent.interpreter.llm is not None:
                    reframed_fact = self.agent.interpreter.verify_and_reframe_fact(
                        original_topic=topic, raw_sentence=first_sentence
                    )
                else:
                    reframed_fact = first_sentence

                if reframed_fact:
                    logger.info(
                        "  [Knowledge Source]: Extracted and verified fact from DuckDuckGo: '%s'",
                        reframed_fact,
                    )
                    return (topic, reframed_fact)

        except requests.exceptions.RequestException as e:
            logger.error(f"   - [DuckDuckGo API]: A network error occurred: {e}")
        except Exception as e:
            logger.error(
                f"   - [DuckDuckGo API]: An unexpected error occurred for topic '{topic}': {e}",
                exc_info=True,
            )

        return None
