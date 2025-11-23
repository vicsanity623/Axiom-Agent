"""
This module defines the core CognitiveAgent class.

It integrates various components like the:
knowledge graph, language parsers, and learning systems to enable complex behaviors such
as conversation, autonomous learning, and self-reflection.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import os
import re
import shutil
import threading
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NotRequired,
    TypedDict,
)

if TYPE_CHECKING:
    from pathlib import Path

from nltk.stem import WordNetLemmatizer

from .coding_agent import CodingAgent
from .config import DEFAULT_BRAIN_FILE, DEFAULT_LLM_PATH, DEFAULT_STATE_FILE
from .constants import CONTRACTION_MAP, RELATION_TYPE_MAP
from .conversation_manager import ConversationManager
from .goal_manager import GoalManager
from .graph_core import ConceptGraph, ConceptNode
from .knowledge_base import seed_core_vocabulary, seed_domain_knowledge
from .knowledge_harvester import KnowledgeHarvester
from .knowledge_manager import KnowledgeManager
from .lexicon_manager import LexiconManager
from .symbolic_parser import SymbolicParser
from .universal_interpreter import (
    InterpretData,
    PropertyData,
    RelationData,
    Task,
    UniversalInterpreter,
)

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()


class ClarificationContext(TypedDict):
    """Hold contextual information needed for a clarification request."""

    subject: NotRequired[str]
    conflicting_relation: NotRequired[str]
    conflicting_nodes: NotRequired[list[str]]


class CognitiveAgent:
    """Orchestrate the primary cognitive functions of the Axiom Agent."""

    INTERPRETER_REBOOT_THRESHOLD: ClassVar[int] = 350
    _EXCLUSIVE_RELATIONS: ClassVar[frozenset[str]] = frozenset(
        ["has_name", "is_capital_of", "has_capital", "is_located_in"]
    )

    def __init__(
        self,
        brain_file: Path = DEFAULT_BRAIN_FILE,
        state_file: Path = DEFAULT_STATE_FILE,
        *,
        load_from_file: bool = True,
        brain_data: dict[str, Any] | None = None,
        cache_data: dict[str, Any] | None = None,
        inference_mode: bool = False,
        enable_llm: bool = True,
    ) -> None:
        """Initialize a new instance of the CognitiveAgent."""
        logger.info("[yellow]Initializing Cognitive Agent...[/yellow]")

        self.brain_file = brain_file
        self.state_file = state_file
        self.inference_mode = inference_mode
        self.interaction_lock = threading.Lock()

        self.graph: ConceptGraph
        self.interpreter: UniversalInterpreter
        self.lexicon: LexiconManager
        self.parser: SymbolicParser
        self.lemmatizer = WordNetLemmatizer()
        self.goal_manager: GoalManager
        self.harvester: KnowledgeHarvester | None = None
        self.learning_goals: list[str | Task] = []
        self.high_priority_tasks: list[Task] = []
        self.pending_relations: dict[str, list[tuple[RelationData, PropertyData]]] = (
            defaultdict(list)
        )
        self.recently_researched: dict[str, float] = {}
        self.learning_iterations: int = 0
        self.autonomous_cycle_count: int = 0
        self.self_node: ConceptNode | None = None
        self.user_node: ConceptNode | None = None
        self.is_awaiting_clarification = False
        self.clarification_context: ClarificationContext = ClarificationContext({})
        self.structured_history: list[tuple[str, list[InterpretData]]] = []

        if self.inference_mode:
            logger.info("   - Running in INFERENCE-ONLY mode. Learning is disabled.")

        self.brain_file.parent.mkdir(parents=True, exist_ok=True)

        self.interpreter = UniversalInterpreter(
            model_path=DEFAULT_LLM_PATH, load_llm=enable_llm
        )
        self.coding_agent = CodingAgent(self.interpreter)  # Initialize Coding Agent
        self.lexicon = LexiconManager(self)
        self.parser = SymbolicParser(self)
        self.goal_manager = GoalManager(self)
        if not self.inference_mode:
            self.harvester = KnowledgeHarvester(agent=self, lock=self.interaction_lock)

        self.knowledge_manager = KnowledgeManager(self)
        self.conversation_manager = ConversationManager(self)

        if load_from_file:
            if not self.brain_file.exists():
                logger.info(
                    f"   - No user brain found at {self.brain_file}. Initializing from master template."
                )
                try:
                    master_brain_ref = importlib.resources.files("axiom").joinpath(
                        "static/master_brain.json"
                    )

                    if master_brain_ref.is_file():
                        with importlib.resources.as_file(
                            master_brain_ref
                        ) as master_brain_path:
                            shutil.copy(master_brain_path, self.brain_file)
                        logger.info(
                            "   - Successfully copied master brain to user directory."
                        )
                    else:
                        logger.warning(
                            "   - Master brain template not found. A fresh brain will be created and seeded."
                        )

                except Exception as e:
                    logger.error(
                        f"   - Failed to copy master brain. A fresh brain will be created. Error: {e}"
                    )

            logger.info(
                f"[green]   - Loading brain from file: {self.brain_file}[/green]"
            )
            self.graph = ConceptGraph.load_from_file(self.brain_file)

        elif brain_data is not None:
            logger.info("   - Initializing brain from provided data dictionary.")
            self.graph = ConceptGraph.load_from_dict(brain_data)
            if cache_data:
                self.interpreter.interpretation_cache = dict(
                    cache_data.get("interpretations", [])
                )
                self.interpreter.synthesis_cache = dict(cache_data.get("synthesis", []))
        else:
            raise ValueError(
                "Agent must be initialized with either files or data if load_from_file is False."
            )

        agent_node = self.graph.get_node_by_name("agent")
        has_name_edge = False
        if agent_node:
            has_name_edge = any(
                edge.type == "has_name"
                for edge in self.graph.get_edges_from_node(agent_node.id)
            )

        if not has_name_edge:
            logger.critical(
                "[red]   - ERROR: Brain is invalid or missing. Re-seeding.[/red]"
            )
            self.graph = ConceptGraph()
            seed_domain_knowledge(self)
            seed_core_vocabulary(self)
            self.save_brain()
            self.save_state()

        self._initialize_identity()

    def chat(self, user_input: str) -> str:
        """Public entry point for user interaction."""
        # Check for Coding Mode command
        if user_input.strip().startswith("/code"):
            return self.coding_agent.process_request(user_input)

        return self.conversation_manager.chat(user_input)

    def _initialize_identity(self):
        """
        Finds or creates the core 'self' and 'user' nodes and assigns them
        to instance attributes for easy access. This is an idempotent operation.
        """
        self_node = self.knowledge_manager.get_or_create_concept(
            "axiom agent", node_type="self_identity"
        )
        if not self_node:
            logger.error("[Identity]: CRITICAL - Failed to create 'axiom agent' node.")
        self.self_node = self_node

        user_node = self.knowledge_manager.get_or_create_concept(
            "current user", node_type="user_identity"
        )
        if not user_node:
            logger.error("[Identity]: CRITICAL - Failed to create 'current user' node.")
        self.user_node = user_node

    def load_state(self) -> None:
        """Load the agent's complete operational state from the state file."""
        if not self.state_file.exists():
            logger.info("   - No agent state file found. Starting with a fresh state.")
            return

        def sanitize_llm_artifacts(obj):
            """Recursively remove LLM instruction tokens from strings in nested structures."""
            if isinstance(obj, str):
                return obj.replace("[/INST]", "").replace("[INST]", "").strip()
            if isinstance(obj, dict):
                return {k: sanitize_llm_artifacts(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize_llm_artifacts(item) for item in obj]
            return obj

        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                state_data = json.load(f)

            # Sanitize all loaded data to prevent Rich markup crashes
            state_data = sanitize_llm_artifacts(state_data)

            self.learning_goals = state_data.get("learning_goals", [])
            self.high_priority_tasks = state_data.get("high_priority_tasks", [])
            self.autonomous_cycle_count = state_data.get("autonomous_cycle_count", 0)
            self.learning_iterations = state_data.get("learning_iterations", 0)
            self.goal_manager.goals = state_data.get("goals", {})
            if self.harvester:
                self.harvester.researched_terms = set(
                    state_data.get("researched_terms", [])
                )
            logger.info(f"   - Agent state loaded from '{self.state_file}'.")
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                f"   - Error loading agent state from {self.state_file}: {e}. Starting fresh."
            )

            self.learning_goals, self.high_priority_tasks = [], []
            self.autonomous_cycle_count, self.learning_iterations = 0, 0
            self.goal_manager.goals = {}

    def save_state(self) -> None:
        """Save the agent's complete operational state to its state file atomically."""
        logger.info(f"   - Saving agent's operational state to '{self.state_file}'...")
        try:
            state_data = {
                "learning_goals": self.learning_goals,
                "high_priority_tasks": self.high_priority_tasks,
                "autonomous_cycle_count": self.autonomous_cycle_count,
                "learning_iterations": self.learning_iterations,
                "researched_terms": list(
                    self.harvester.researched_terms if self.harvester else []
                ),
                "goals": self.goal_manager.goals,
            }

            # Atomic write: write to temp file then rename
            temp_file = self.state_file.with_suffix(".tmp")
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=4)

            os.replace(temp_file, self.state_file)
            logger.info("   - Agent state saved successfully.")
        except Exception as e:
            logger.error(f"   - CRITICAL ERROR: Failed to save agent state. Error: {e}")

    def save_brain(self) -> None:
        """Save the current knowledge graph to its JSON file atomically."""
        if self.inference_mode:
            return
        try:
            # Atomic write
            temp_file = self.brain_file.with_suffix(".tmp")
            self.graph.save_to_file(temp_file)
            os.replace(temp_file, self.brain_file)
            logger.info(f"[cyan]   - Agent brain saved to {self.brain_file}[/cyan]")
        except Exception as e:
            logger.error(f"   - CRITICAL ERROR: Failed to save brain. Error: {e}")

    def log_autonomous_cycle_completion(self) -> None:
        """Increment the autonomous cycle counter and trigger an interpreter reboot if needed."""
        self.autonomous_cycle_count += 1
        if self.autonomous_cycle_count >= self.INTERPRETER_REBOOT_THRESHOLD:
            self._reboot_interpreter()
            self.autonomous_cycle_count = 0

    def _reboot_interpreter(self) -> None:
        """Destroy and recreate the UniversalInterpreter to ensure stability."""
        logger.warning(
            "\n--- [SYSTEM HEALTH]: Prophylactically rebooting Universal Interpreter ---"
        )
        logger.info("This is a preventative measure to ensure long-term stability.")
        old_interp_cache = self.interpreter.interpretation_cache
        old_synth_cache = self.interpreter.synthesis_cache
        del self.interpreter
        self.interpreter = UniversalInterpreter()
        self.interpreter.interpretation_cache = old_interp_cache
        self.interpreter.synthesis_cache = old_synth_cache
        logger.info(
            "--- [SYSTEM HEALTH]: Interpreter reboot complete. Caches restored. ---\n"
        )

    def _clean_phrase(self, phrase: str) -> str:
        """
        Clean and normalize a phrase for use as a concept or relation in the graph.
        This version now includes lemmatization for better root word identification.
        """
        digit_map = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        clean_phrase = phrase.lower().strip().replace("_", " ")
        clean_phrase = re.sub(r"[^a-z0-9\s-]", "", clean_phrase)

        processed_words = []
        for word in clean_phrase.split():
            # First, filter out malformed words (like 'ch2ch')
            if not (word.isalpha() or word.isdigit() or "-" in word):
                continue

            # Second, lemmatize words ending in 's' to handle plurals/possessives
            if word.endswith("s") and len(word) > 1:
                # Use the noun lemmatizer to convert e.g., "cats" -> "cat", "someones" -> "someone"
                lemmatized_word = self.lemmatizer.lemmatize(word, pos="n")
                processed_words.append(lemmatized_word)
            else:
                processed_words.append(word)

        # Third, convert any remaining digits to words
        normalized_words = [digit_map.get(word, word) for word in processed_words]
        clean_phrase = " ".join(normalized_words)

        # The rest of your logic remains the same
        clean_phrase = re.sub(r"\s*\([^)]*\)", "", clean_phrase).strip()
        if clean_phrase.endswith("'s"):
            clean_phrase = clean_phrase[:-2].strip()

        words = clean_phrase.split()
        if len(words) == 2 and words[0] in ("a", "an", "the"):
            clean_phrase = words[1]

        return " ".join(clean_phrase.split())

    def _extract_name_from_relation(
        self, value: str | dict[str, Any] | list | None
    ) -> str | None:
        """Extracts a clean name string from a relation's subject or object."""
        if value is None:
            return None
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str) and (name_clean := name.strip()):
                return self._clean_phrase(name_clean)
            return None
        if name_clean := value.strip():
            return self._clean_phrase(name_clean) or None
        return None

    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        words = text.lower().split()
        expanded_words = [CONTRACTION_MAP.get(word, word) for word in words]
        expanded_text = " ".join(expanded_words)
        if expanded_text != text.lower():
            logger.info(
                f"  [Contraction Expander]: Normalized input to '{expanded_text}'"
            )
        return expanded_text

    def _preprocess_self_reference(self, text: str) -> str:
        """Normalize user input to replace self-references with a canonical name."""
        processed_text = re.sub(
            r"\byour name\b", "the agent's name", text, flags=re.IGNORECASE
        )
        processed_text = re.sub(
            r"\bwho are you\b", "what is the agent", processed_text, flags=re.IGNORECASE
        )
        processed_text = re.sub(
            r"\byou are\b", "the agent is", processed_text, flags=re.IGNORECASE
        )
        processed_text = re.sub(
            r"(?<!thank you for )\byour\b",
            "the agent's",
            processed_text,
            flags=re.IGNORECASE,
        )
        processed_text = re.sub(
            r"(?<!thank )\byou\b", "the agent", processed_text, flags=re.IGNORECASE
        )
        if processed_text != text:
            logger.debug(f"  [Pre-processor]: Normalized input to '{processed_text}'")
        return processed_text

    def get_relation_type(self, verb: str, subject: str, object_: str) -> str:
        """Determine the semantic relationship type from a simple verb."""
        if "agent" in subject.lower() and verb in ["be", "is", "are", "is named"]:
            if len(object_.split()) == 1 and object_[0].isupper():
                return "has name"
        return RELATION_TYPE_MAP.get(verb, verb)
