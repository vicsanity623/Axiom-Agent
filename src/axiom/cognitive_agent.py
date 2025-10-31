from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import date, datetime
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    NotRequired,
    TypedDict,
    cast,
)

from thefuzz import fuzz

if TYPE_CHECKING:
    from pathlib import Path

from nltk.stem import WordNetLemmatizer
from thefuzz import process

from .config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE
from .dictionary_utils import get_word_info_from_wordnet
from .goal_manager import GoalManager
from .graph_core import ConceptGraph, ConceptNode, RelationshipEdge
from .knowledge_base import (
    seed_core_vocabulary,
    seed_domain_knowledge,
    validate_and_add_relation,
)
from .knowledge_harvester import KnowledgeHarvester
from .lexicon_manager import LexiconManager
from .symbolic_parser import SymbolicParser
from .universal_interpreter import (
    Entity,
    InterpretData,
    RelationData,
    UniversalInterpreter,
)

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()


class ClarificationContext(TypedDict):
    """Hold contextual information needed for a clarification request."""

    subject: NotRequired[str]
    conflicting_relation: NotRequired[str]
    conflicting_nodes: NotRequired[list[str]]


RELATION_TYPE_MAP: Final[dict[str, str]] = {
    "be": "is_a",
    "is": "is_a",
    "are": "is_a",
    "cause": "causes",
    "causes": "causes",
    "locate_in": "is_located_in",
    "located_in": "is_located_in",
    "part_of": "is_part_of",
    "learn": "learns",
    "release": "released",
    "released": "released",
}

CONTRACTION_MAP: Final[dict[str, str]] = {
    "what's": "what is",
    "whats": "what is",
    "it's": "it is",
    "its": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
}

PROVENANCE_RANK: Final[dict[str, int]] = {
    "seed": 5,
    "dictionary_api": 4,
    "llm_verified": 3,
    "wikipedia": 2,
    "duckduckgo": 2,
    "user_input": 1,
    "llm": 1,
}


class CognitiveAgent:
    """Orchestrate the primary cognitive functions of the Axiom Agent."""

    INTERPRETER_REBOOT_THRESHOLD: ClassVar[int] = 350

    _EXCLUSIVE_RELATIONS: ClassVar[frozenset[str]] = frozenset(
        [
            "has_name",
            "is_capital_of",
            "has_capital",
            "is_located_in",
        ],
    )

    def __init__(
        self,
        brain_file: Path = DEFAULT_BRAIN_FILE,
        state_file: Path = DEFAULT_STATE_FILE,
        *,
        load_from_file: bool = True,
        brain_data: dict[str, object] | None = None,
        cache_data: dict[str, object] | None = None,
        inference_mode: bool = False,
        enable_llm: bool = True,
    ) -> None:
        """Initialize a new instance of the CognitiveAgent.

        This constructor sets up the agent's core components and loads its cognitive
        state (the "brain") from either a file or provided data dictionaries. It
        also performs an integrity check on the loaded brain, re-seeding it with
        foundational knowledge if core identity information is missing.

        Args:
            brain_file: Path to the file for storing the concept graph.
            state_file: Path to the file for storing the agent's operational state.
            load_from_file: If True, loads the brain and state from the specified
                files. If False, `brain_data` and `cache_data` must be provided.
            brain_data: A dictionary containing the serialized concept graph and
                metadata. Used when `load_from_file` is False.
            cache_data: A dictionary containing serialized interpretation and
                synthesis caches. Used when `load_from_file` is False.
            inference_mode: If True, disables learning and knowledge harvesting.
            enable_llm: If True, enables the Large Language Model backend for the
                UniversalInterpreter.

        Raises:
            ValueError: If the agent is not initialized with either files or data.
            TypeError: If `brain_data` or `cache_data` have malformed contents.
        """
        logger.info("Initializing Cognitive Agent...")
        self.brain_file = brain_file
        self.state_file = state_file
        self.inference_mode = inference_mode
        if self.inference_mode:
            logger.info("   - Running in INFERENCE-ONLY mode. Learning is disabled.")

        # Initialize core components
        self.brain_file.parent.mkdir(parents=True, exist_ok=True)
        self.interaction_lock = threading.Lock()
        self.interpreter = UniversalInterpreter(load_llm=enable_llm)
        self.lexicon: LexiconManager = LexiconManager(self)
        self.parser: SymbolicParser = SymbolicParser(self)
        self.lemmatizer = WordNetLemmatizer()
        self.goal_manager: GoalManager = GoalManager(self)
        self.learning_goals: list[str | dict[str, object]] = []
        self.pending_relations: list[tuple[RelationData, dict[str, object], float]] = []
        self.recently_researched: dict[str, float] = {}
        self.harvester: KnowledgeHarvester | None = None
        if not self.inference_mode:
            self.harvester = KnowledgeHarvester(agent=self, lock=self.interaction_lock)

        # Load and verify brain based on initialization mode
        if load_from_file:
            logger.info(
                "[green]   - Loading brain from file: %s[/green]", self.brain_file
            )
            self.graph = ConceptGraph.load_from_file(self.brain_file)
            self._load_agent_state()

            # Ensure brain integrity by checking for core agent identity
            agent_node = self.graph.get_node_by_name("agent")
            has_name_edge = False
            if agent_node:
                has_name_edge = any(
                    edge.type == "has_name"
                    for edge in self.graph.get_edges_from_node(agent_node.id)
                )

            if not has_name_edge:
                logger.critical(
                    "   - CRITICAL FAILURE: Missing identity, Re-seeding brain for integrity."
                )
                self.graph = ConceptGraph()
                seed_domain_knowledge(self)
                seed_core_vocabulary(self)
                self.save_brain()
                self.save_state()

        elif brain_data is not None and cache_data is not None:
            logger.info("   - Initializing brain from loaded .axm model data.")
            self.graph = ConceptGraph.load_from_dict(brain_data)

            # Type-safe loading of cache data
            interpretations_raw = cache_data.get("interpretations", [])
            synthesis_raw = cache_data.get("synthesis", [])
            if not isinstance(interpretations_raw, list) or not isinstance(
                synthesis_raw, list
            ):
                raise TypeError(
                    "Cache data for 'interpretations' and 'synthesis' must be lists."
                )
            self.interpreter.interpretation_cache = dict(interpretations_raw)
            self.interpreter.synthesis_cache = dict(synthesis_raw)

            # Type-safe loading of other brain data
            learning_iterations_raw = brain_data.get("learning_iterations", 0)
            if not isinstance(learning_iterations_raw, int):
                raise TypeError("'learning_iterations' must be an integer.")
            self.learning_iterations = learning_iterations_raw
        else:
            raise ValueError("Agent must be initialized with either files or data.")

        # Initialize operational state
        self.is_awaiting_clarification = False
        self.clarification_context: ClarificationContext = ClarificationContext({})
        self.structured_history: list[tuple[str, list[InterpretData]]] = []
        self.autonomous_cycle_count = 0

    def chat(self, user_input: str) -> str:
        """
        Process a single user input and return the agent's response by orchestrating
        a multi-stage cognitive pipeline.
        """
        logger.info(" \nUser: %s", user_input)
        self.graph.decay_activations()

        if self.is_awaiting_clarification:
            return self._handle_clarification(user_input)

        interpretations = self._get_interpretation(user_input)

        if not interpretations:
            reflex_response = self._handle_cognitive_reflex(user_input)
            if reflex_response:
                return reflex_response

            return "I'm sorry, I was unable to understand that."

        self._update_history(interpretations)
        structured_response = self._process_intent(interpretations[0], user_input)

        if len(interpretations) > 1:
            self._learn_from_extra_clauses(interpretations[1:])

        final_response, synthesizer_was_used = self._synthesize_response(
            structured_response, user_input
        )

        if synthesizer_was_used and not self.inference_mode:
            self._learn_from_introspection(final_response, interpretations[0])

        return final_response

    def _get_interpretation(self, user_input: str) -> list[InterpretData] | None:
        """Pipeline for parsing and interpreting user input."""
        sanitized_input = self._sanitize_sentence_for_learning(user_input)
        expanded_input = self._expand_contractions(sanitized_input)
        contextual_input = self._resolve_references(expanded_input)
        normalized_input = self._preprocess_self_reference(contextual_input)

        interpretations = self.parser.parse(normalized_input)

        if not interpretations:
            logger.warning(
                "  [Cognitive Flow]: Symbolic parsing failed. Falling back to LLM interpreter."
            )
            llm_interpretation = self.interpreter.interpret(normalized_input)
            if llm_interpretation:
                return [llm_interpretation]

        return interpretations

    def _handle_cognitive_reflex(self, normalized_input: str) -> str | None:
        """Handle cases where interpretation fails due to unknown words."""
        words = normalized_input.lower().split()
        unknown_words = [w for w in words if w and not self.lexicon.is_known_word(w)]

        if not unknown_words:
            return None

        word_to_learn = sorted(set(unknown_words))[0]
        goal = f"INVESTIGATE: {word_to_learn}"
        self.learning_goals.append(goal)
        logger.info(
            "  [Cognitive Reflex]: Unknown word '%s', added to learning goals.",
            word_to_learn,
        )

        if self.harvester and not self.inference_mode:
            was_resolved = self.harvester._resolve_investigation_goal(goal)
            if was_resolved and self.lexicon.is_known_word(word_to_learn):
                return self._chat_reentry_once(normalized_input)
            return f"I discovered a new word, '{word_to_learn}', but my attempt to research it in real-time failed. I will study it later."

        return f"New word '{word_to_learn}' discovered. I must study it before I can understand."

    def _update_history(self, interpretations: list[InterpretData]) -> None:
        """Updates the conversational history."""
        self.structured_history.append(("user", interpretations))
        if len(self.structured_history) > 10:
            self.structured_history = self.structured_history[-10:]

    def _learn_from_extra_clauses(
        self, extra_interpretations: list[InterpretData]
    ) -> None:
        """Processes additional factual clauses from a multi-clause input."""
        for extra in extra_interpretations:
            if extra.get("intent") == "statement_of_fact":
                if rel := extra.get("relation"):
                    self._process_statement_for_learning(rel)

    def _learn_from_introspection(
        self, final_response: str, original_interpretation: InterpretData
    ) -> None:
        """Analyzes the agent's own synthesized response to learn new facts."""
        intent = original_interpretation.get("intent", "")
        context_subject: str | None = None
        if intent.startswith("question"):
            if relation := original_interpretation.get("relation"):
                subject_raw = relation.get("subject")
                if isinstance(subject_raw, str):
                    context_subject = subject_raw
            elif entities := original_interpretation.get("entities"):
                name_raw = entities[0].get("name")
                if isinstance(name_raw, str):
                    context_subject = name_raw

        if context_subject:
            logger.info(
                "  [Introspection]: Analyzing synthesized response for new knowledge..."
            )
            new_interpretations = self.parser.parse(
                final_response, context_subject=context_subject
            )
            if new_interpretations:
                self._learn_from_extra_clauses(new_interpretations)

    def _chat_reentry_once(self, user_input: str) -> str:
        """Safely re-enter chat() once after learning, avoiding infinite recursion."""
        if getattr(self, "_has_reentered_chat", False):
            logger.warning("  [Safety]: Prevented recursive chat() re-entry.")
            return "I've already reconsidered that input after learning something new."

        self._has_reentered_chat = True
        try:
            return self.chat(user_input)
        finally:
            self._has_reentered_chat = False

    def _resolve_references(self, text: str) -> str:
        """Resolve simple pronouns using the stored interpretations from history."""
        pronouns_to_resolve = {"it", "they", "its", "their", "them"}
        if not any(
            re.search(rf"\b{pronoun}\b", text, re.IGNORECASE)
            for pronoun in pronouns_to_resolve
        ):
            logger.debug("[Coreference]: No pronouns found in text.")
            return text

        antecedent_raw: str | dict[str, Any] | list | None = None
        for speaker, interpretations in reversed(self.structured_history):
            if speaker == "user" and interpretations:
                primary_interpretation = interpretations[0]
                relation = primary_interpretation.get("relation")
                entities = primary_interpretation.get("entities")

                if relation and relation.get("subject"):
                    antecedent_raw = relation["subject"]
                    break
                if entities:
                    antecedent_raw = entities[0]["name"]
                    break

        if antecedent_raw:
            antecedent: str | None = None
            if isinstance(antecedent_raw, str):
                antecedent = antecedent_raw
            elif isinstance(antecedent_raw, dict):
                antecedent = antecedent_raw.get("name")

            if antecedent:
                clean_antecedent = self._clean_phrase(antecedent)
                if not clean_antecedent:
                    logger.debug(
                        "[Coreference]: Found antecedent %s, but it was empty after cleaning.",
                        antecedent,
                    )
                    return text

                modified_text = re.sub(
                    r"\b(it|they|them)\b",
                    clean_antecedent,
                    text,
                    flags=re.IGNORECASE,
                )
                modified_text = re.sub(
                    r"\b(its|their)\b",
                    f"{clean_antecedent}'s",
                    modified_text,
                    flags=re.IGNORECASE,
                )

                if modified_text != text:
                    logger.info(
                        "  [Coreference]: Resolved pronouns, transforming '%s' to '%s'",
                        text,
                        modified_text,
                    )
                    return modified_text
                logger.debug(
                    "[Coreference]: No substitution occurred despite antecedent found.",
                )
        else:
            logger.debug("[Coreference]: No antecedent found for pronoun resolution.")

        return text

    def _handle_clarification(self, user_input: str) -> str:
        """Handle the user's response after a contradiction was detected."""
        logger.info("  [Curiosity]: Processing user's clarification...")
        interpretation = self.interpreter.interpret(user_input)

        entities = interpretation.get("entities", [])

        if not entities:
            logger.warning(
                "  [Curiosity]: No entities found in user's clarification. No update applied.",
            )
            return "I'm sorry, I couldn't understand your clarification."

        entity_name_raw = entities[0].get("name")
        if not isinstance(entity_name_raw, str):
            return (
                "I'm sorry, I couldn't identify a clear concept in your clarification."
            )

        correct_answer_name = self._clean_phrase(entity_name_raw)
        subject_name: str | None = self.clarification_context.get("subject")
        relation_type = self.clarification_context.get("conflicting_relation")

        if not subject_name or not relation_type:
            logger.warning(
                "  [Curiosity]: Clarification context incomplete (subject=%s, relation=%s).",
                subject_name,
                relation_type,
            )
            return "I'm sorry, I cannot process this clarification."

        subject_node = self.graph.get_node_by_name(subject_name)
        if not subject_node:
            logger.warning(
                "  [Curiosity]: Subject node '%s' not found in graph. Creating it.",
                subject_name,
            )
            subject_node = ConceptNode(name=subject_name)
            self.graph.add_node(subject_node)

        correct_node = self.graph.get_node_by_name(correct_answer_name)
        if not correct_node:
            logger.info(
                "  [Curiosity]: Creating node for correct answer '%s'.",
                correct_answer_name,
            )
            correct_node = ConceptNode(name=correct_answer_name)
            self.graph.add_node(correct_node)

        self._gather_facts_multihop.cache_clear()
        logger.info("  [Cache]: Cleared reasoning cache due to knowledge correction.")

        updated_any = False
        for u, v, key, data in list(
            self.graph.graph.out_edges(subject_node.id, keys=True, data=True),
        ):
            if data.get("type") == relation_type:
                target_node = self.graph.get_node_by_id(v)
                if (
                    target_node
                    and self._clean_phrase(target_node.name) == correct_answer_name
                ):
                    self.graph.graph[u][v][key]["weight"] = 1.0
                    logger.info(
                        "    - REINFORCED: %s --[%s]--> %s",
                        subject_name,
                        relation_type,
                        correct_answer_name,
                    )
                    updated_any = True
                else:
                    self.graph.graph[u][v][key]["weight"] = 0.1
                    if target_node:
                        logger.info(
                            "    - PUNISHED: %s --[%s]--> %s",
                            subject_name,
                            relation_type,
                            target_node.name,
                        )
                    updated_any = True

        if not any(
            (node := self.graph.get_node_by_id(v))
            and self._clean_phrase(node.name) == correct_answer_name
            and data.get("type") == relation_type
            for u, v, key, data in self.graph.graph.out_edges(
                subject_node.id,
                keys=True,
                data=True,
            )
        ):
            logger.info(
                "  [Curiosity]: Adding missing edge for clarification: %s --[%s]--> %s",
                subject_name,
                relation_type,
                correct_answer_name,
            )
            self.graph.add_edge(subject_node, correct_node, relation_type, weight=1.0)
            updated_any = True

        if updated_any:
            self.save_brain()
            self.is_awaiting_clarification = False
            self.clarification_context = {}
            return "Thank you for the clarification. I have updated my knowledge."
        logger.warning(
            "  [Curiosity]: No updates were applied despite clarification.",
        )
        return "I could not update my knowledge from your clarification."

    def _get_corrected_entity(self, entity_name: str) -> str:
        """Uses fuzzy matching to correct potential typos in an entity name."""
        all_concepts = list(self.graph.get_all_node_names())
        if not all_concepts:
            return entity_name

        match_result = process.extractOne(entity_name, all_concepts)
        if not match_result:
            return entity_name

        best_match: str = match_result[0]
        score: int = match_result[1]

        if score > 85:
            if entity_name != best_match:
                logger.info(
                    "  [Cognitive Reflex]: Corrected entity '%s' to '%s' (confidence: %d%%).",
                    entity_name,
                    best_match,
                    score,
                )
            return best_match

        return entity_name

    def _process_intent(
        self,
        interpretation: InterpretData,
        user_input: str,
    ) -> str:
        """Route the interpreted user input to the appropriate cognitive function."""
        intent = interpretation.get("intent", "unknown")
        entities: list[Entity] = interpretation.get("entities", [])
        relation: RelationData | None = interpretation.get("relation")

        if intent == "greeting":
            return "Hello User."
        if intent == "farewell":
            return "Goodbye User."
        if intent in ("gratitude", "acknowledgment"):
            return "You're welcome!"
        if intent == "positive_affirmation":
            return "I'm glad you think so!"

        if intent == "meta_question_self":
            response: str | None = self._answer_question_about("agent", user_input)
            return (
                response
                if response is not None
                else "I am a cognitive agent designed to learn and assist users."
            )

        if intent == "meta_question_purpose":
            response = self._find_specific_fact("agent", "has_purpose")
            return (
                response
                if response is not None
                else "I am an AI assistant designed to learn and help users."
            )

        if intent == "meta_question_abilities":
            response = self._find_specific_fact("agent", "has_ability")
            return (
                response
                if response is not None
                else "I can learn new facts, answer questions, and reason about information."
            )

        if intent == "command_show_all_facts":
            return self._get_all_facts_as_string()

        if intent == "statement_of_fact" and relation:
            was_learned, learn_msg = self._process_statement_for_learning(relation)
            logger.info(
                "  [Knowledge Acquisition]: Learned = '%s', msg = '%s'",
                was_learned,
                learn_msg,
            )
            if was_learned:
                return "I understand. I have noted that."
            if learn_msg == "exclusive_conflict":
                pass
            return f"I tried to record that fact but something went wrong: {learn_msg}"

        if intent == "question_yes_no" and relation:
            return self._answer_yes_no_question(relation)

        if intent == "question_by_relation" and relation:
            subject_raw = relation.get("subject")
            subject_name = self._extract_name_from_relation(subject_raw)

            verb = relation.get("verb")

            if subject_name and verb:
                corrected_subject = self._get_corrected_entity(subject_name)
                subject_node = self.graph.get_node_by_name(corrected_subject)

                if subject_node:
                    for edge in self.graph.get_edges_from_node(subject_node.id):
                        if edge.type == verb:
                            target_node = self.graph.get_node_by_id(edge.target)
                            if target_node:
                                property_name = verb.replace("has_", "").replace(
                                    "_",
                                    " ",
                                )
                                return f"The {property_name} of {subject_name.capitalize()} is {target_node.name.capitalize()}."

            return f"I don't have information about the {relation.get('verb', 'property').replace('has_', '')} of {subject_name or 'that'}."

        if intent in ("question_about_entity", "question_about_concept"):
            if entities:
                entity_name = entities[0]["name"]
                corrected_entity_name = self._get_corrected_entity(entity_name)
                response = self._answer_question_about(
                    corrected_entity_name, user_input
                )
                return (
                    response
                    if response is not None
                    else f"I don't have any specific information about '{corrected_entity_name}' right now."
                )

        return "I'm not sure how to process that. Could you rephrase?"

    def _find_specific_fact(self, subject_name: str, relation_type: str) -> str | None:
        """Finds all facts of a specific type related to a subject and formats them."""
        subject_node = self.graph.get_node_by_name(subject_name)
        if not subject_node:
            return None

        edges = self.graph.get_edges_from_node(subject_node.id)

        relevant_edges = [edge for edge in edges if edge.type == relation_type]

        if not relevant_edges:
            return None

        object_names = []
        for edge in relevant_edges:
            target_node = self.graph.get_node_by_id(edge.target)
            if target_node:
                object_names.append(target_node.name)

        if not object_names:
            return None

        if len(object_names) == 1:
            return f"My {relation_type.replace('_', ' ')} is to {object_names[0]}."
        formatted_list = ", ".join(object_names[:-1]) + f", and {object_names[-1]}"
        return f"My abilities include: {formatted_list}."

    def _get_all_facts_as_string(self) -> str:
        """Retrieve, filter, and format all facts from the knowledge graph."""
        all_facts = []
        all_edges = self.graph.get_all_edges()

        if not all_edges:
            return "My knowledge base is currently empty."

        high_confidence_edges = sorted(
            [
                edge
                for edge in all_edges
                if edge.type != "might_relate" and edge.weight >= 0.8
            ],
            key=lambda edge: edge.weight,
            reverse=True,
        )

        for edge in high_confidence_edges:
            source_node = self.graph.get_node_by_id(edge.source)
            target_node = self.graph.get_node_by_id(edge.target)

            if source_node and target_node:
                fact_string = (
                    f"- {source_node.name.capitalize()} "
                    f"--[{edge.type}]--> "
                    f"{target_node.name.capitalize()} "
                    f"(Weight: {edge.weight:.2f})"
                )
                all_facts.append(fact_string)

        if all_facts:
            return (
                "Here are all the high-confidence facts I know (strongest first):\n\n"
                + "\n".join(all_facts)
            )

        return "I currently lack high-confidence facts to display."

    def _answer_question_about(self, entity_name: str, user_input: str) -> str:
        """Find and return known facts related to a specific entity."""
        clean_entity_name = self._clean_phrase(entity_name)
        if not clean_entity_name:
            return f"I don't have any information about {entity_name}."

        if "agent" in clean_entity_name and "name" in user_input.lower():
            agent_node = self.graph.get_node_by_name("agent")
            if agent_node:
                name_edge = next(
                    (
                        edge
                        for edge in self.graph.get_edges_from_node(agent_node.id)
                        if edge.type == "has_name"
                    ),
                    None,
                )
                if name_edge:
                    name_node = self.graph.get_node_by_id(name_edge.target)
                    if name_node:
                        return f"My name is {name_node.name.capitalize()}."
                    return (
                        "I know I have a name, but I can't seem to recall it right now."
                    )
                return "I don't have a name yet."
            return "I don't seem to have a concept of myself right now."

        subject_node = self.graph.get_node_by_name(clean_entity_name)
        if not subject_node:
            return f"I don't have any information about {entity_name}."

        logger.info(
            "  [CognitiveAgent]: Starting single-hop reasoning for '%s'.",
            entity_name,
        )
        facts_with_props = self._gather_facts_multihop(subject_node.id, max_hops=4)

        is_temporal_query = any(
            keyword in user_input.lower()
            for keyword in ["now", "currently", "today", "this year"]
        )
        if is_temporal_query:
            facts = self._filter_facts_for_temporal_query(facts_with_props)
        else:
            facts = {fact_str for fact_str, _ in facts_with_props}

        if not facts:
            return f"I dont have any details for {subject_node.name.capitalize()}."

        return ". ".join(sorted(facts)) + "."

    def _synthesize_response(
        self,
        structured_response: str,
        user_input: str,
    ) -> tuple[str, bool]:
        """Convert a structured, internal response into natural language."""
        non_synthesize_triggers = [
            "Hello User",
            "Goodbye User",
            "I understand. I have noted that.",
            "I don't have any information about",
            "My name is",
            "I know about",
            "That's an interesting topic about",
            "I'm not sure I fully understood that",
            "You're welcome!",
            "I'm glad you think so!",
            "Here are all the high-confidence facts",
            "Thank you for the clarification.",
            "Thank you. I have corrected my knowledge.",
            "I am currently in a read-only mode",
            "I'm not familiar with the term",
        ]

        if any(trigger in structured_response for trigger in non_synthesize_triggers):
            return (structured_response, False)

        if re.match(r"^\s*(FACT:|RELATION\()", structured_response):
            logger.debug(
                "  [Synthesizer]: Structured input detected → %s",
                structured_response,
            )
            fluent_response = self.interpreter.synthesize(
                structured_response,
                original_question=user_input,
            )
            logger.debug(
                "  [Synthesizer]: Output → %s",
                fluent_response,
            )
            return (fluent_response, True)

        logger.debug(
            "  [Structured Response]: %s",
            structured_response,
        )
        fluent_response = self.interpreter.synthesize(
            structured_response,
            original_question=user_input,
        )
        logger.debug(
            "  [Synthesized Response]: %s",
            fluent_response,
        )
        return (fluent_response, True)

    def _perform_multi_hop_query(
        self,
        start_node: ConceptNode,
        end_node: ConceptNode,
        max_hops: int = 3,
    ) -> list[RelationshipEdge] | None:
        """Find a path of relationships between a start and end node."""
        queue: list[tuple[str, list[RelationshipEdge]]] = [(start_node.id, [])]
        visited: set[str] = {start_node.id}

        while queue:
            current_node_id, path = queue.pop(0)

            if len(path) >= max_hops:
                continue

            for edge in self.graph.get_edges_from_node(current_node_id):
                neighbor_id = edge.target

                if neighbor_id == end_node.id:
                    return path + [edge]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [edge]
                    queue.append((neighbor_id, new_path))

        logger.warning("    [BFS Engine]: FAILED. Queue is empty, no path found.")
        return None

    def _format_path_as_sentence(self, path: list[RelationshipEdge]) -> str:
        """Convert a path of edges into a human-readable sentence."""
        if not path:
            return ""

        parts = []
        for i, edge in enumerate(path):
            source_node = self.graph.get_node_by_id(edge.source)
            target_node = self.graph.get_node_by_id(edge.target)

            if not source_node or not target_node:
                continue

            if i == 0:
                parts.append(
                    f"{source_node.name.capitalize()} {edge.type.replace('_', ' ')} {target_node.name}",
                )
            else:
                parts.append(
                    f"which in turn {edge.type.replace('_', ' ')} {target_node.name}",
                )

        return ", and ".join(parts) + "."

    def _answer_yes_no_question(self, relation: RelationData) -> str:
        """Answer a yes/no question by checking for facts and contradictions."""
        subject_raw = relation.get("subject")
        object_raw = relation.get("object")

        if not isinstance(subject_raw, str) or not isinstance(object_raw, str):
            return "I'm not sure how to answer that."

        subject = subject_raw
        object_ = object_raw

        subject_node = self.graph.get_node_by_name(subject)
        if not subject_node:
            return f"I don't have any information about {subject}."

        for edge in self.graph.get_edges_from_node(subject_node.id):
            target_node = self.graph.get_node_by_id(edge.target)
            if not target_node:
                continue

            if target_node.name == object_:
                return f"Yes, based on what I know, {subject} is {object_}."

            if edge.type in ["is_a", "has_property"]:
                if target_node.name != object_:
                    return f"No, based on what I know, {subject} is {target_node.name}, not {object_}."

        return f"I'm not sure. I don't have any information about whether {subject} is {object_}."

    def _reboot_interpreter(self) -> None:
        """Destroy and recreate the UniversalInterpreter to ensure stability."""
        logger.warning(
            "\n--- [SYSTEM HEALTH]: Prophylactically rebooting Universal Interpreter ---",
        )
        logger.info("This is a preventative measure to ensure long-term stability.")

        old_interp_cache = self.interpreter.interpretation_cache
        old_synth_cache = self.interpreter.synthesis_cache

        del self.interpreter
        self.interpreter = UniversalInterpreter()

        self.interpreter.interpretation_cache = old_interp_cache
        self.interpreter.synthesis_cache = old_synth_cache

        logger.info(
            "--- [SYSTEM HEALTH]: Interpreter reboot complete. Caches restored. ---\n",
        )

    def log_autonomous_cycle_completion(self) -> None:
        """Increment the autonomous cycle counter and trigger an interpreter reboot if needed."""
        self.autonomous_cycle_count += 1
        logger.debug(
            "  [System Health]: Autonomous cycles since last interpreter reboot: %s/%s",
            self.autonomous_cycle_count,
            self.INTERPRETER_REBOOT_THRESHOLD,
        )
        if self.autonomous_cycle_count >= self.INTERPRETER_REBOOT_THRESHOLD:
            self._reboot_interpreter()
            self.autonomous_cycle_count = 0

    def _load_agent_state(self) -> None:
        """Load the agent's operational state from its state file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    state_data = json.load(f)
                    self.learning_iterations = state_data.get("learning_iterations", 0)
                logger.info(
                    "   - Successfully loaded agent state from '%s'.",
                    self.state_file,
                )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "Could not load or parse agent state file '%s'. Resetting state. Error: %s",
                    self.state_file,
                    e,
                )
                self.learning_iterations = 0
        else:
            self.learning_iterations = 0

    def _save_agent_state(self) -> None:
        """Save the agent's current operational state to its state file."""
        state_data = {"learning_iterations": self.learning_iterations}
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=4)

    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions (e.g., "what's" -> "what is")."""
        words = text.lower().split()
        expanded_words = [CONTRACTION_MAP.get(word, word) for word in words]
        expanded_text = " ".join(expanded_words)

        if expanded_text != text.lower():
            logger.info(
                "  [Contraction Expander]: Normalized input to '%s'",
                expanded_text,
            )

        return expanded_text

    def _preprocess_self_reference(self, text: str) -> str:
        """Normalize user input to replace self-references with a canonical name."""
        processed_text = re.sub(
            r"\byour name\b",
            "the agent's name",
            text,
            flags=re.IGNORECASE,
        )
        processed_text = re.sub(
            r"\bwho are you\b",
            "what is the agent",
            processed_text,
            flags=re.IGNORECASE,
        )
        processed_text = re.sub(
            r"\byou are\b",
            "the agent is",
            processed_text,
            flags=re.IGNORECASE,
        )
        processed_text = re.sub(
            r"(?<!thank you for )\byour\b",
            "the agent's",
            processed_text,
            flags=re.IGNORECASE,
        )
        processed_text = re.sub(
            r"(?<!thank )\byou\b",
            "the agent",
            processed_text,
            flags=re.IGNORECASE,
        )
        if processed_text != text:
            logger.debug(
                "  [Pre-processor]: Normalized input to '%s'",
                processed_text,
            )
        return processed_text

    @lru_cache(maxsize=256)
    def _gather_facts_multihop(
        self,
        start_node_id: str,
        max_hops: int,
    ) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
        """Gather all facts related to a starting node via graph traversal."""
        logger.info(
            "  [Cache]: MISS! Executing full multi-hop graph traversal for node ID: %s",
            start_node_id,
        )
        start_node = self.graph.get_node_by_id(start_node_id)
        if not start_node:
            return ()

        found_facts: dict[str, RelationshipEdge] = {}
        queue: list[tuple[str, int]] = [(start_node_id, 0)]
        visited: set[str] = {start_node_id}

        while queue:
            current_node_id, current_hop = queue.pop(0)
            if current_hop >= max_hops:
                continue

            current_node = self.graph.get_node_by_id(current_node_id)
            if not current_node:
                continue

            for edge in self.graph.get_edges_from_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                target_node = self.graph.get_node_by_id(edge.target)
                if target_node:
                    fact_str = f"{current_node.name} {edge.type.replace('_', ' ')} {target_node.name}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append((edge.target, current_hop + 1))
            for edge in self.graph.get_edges_to_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                source_node = self.graph.get_node_by_id(edge.source)
                if source_node:
                    fact_str = f"{source_node.name} {edge.type.replace('_', ' ')} {current_node.name}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.source not in visited:
                        visited.add(edge.source)
                        queue.append((edge.source, current_hop + 1))

        all_facts_items = list(found_facts.items())

        if len(all_facts_items) > 10:
            original_subject = start_node.name
            relevance_filtered = [
                (f, e)
                for f, e in all_facts_items
                if f.lower().startswith(original_subject)
            ]
            if relevance_filtered:
                all_facts_items = relevance_filtered
        if len(all_facts_items) > 10:
            all_facts_items.sort(key=lambda item: item[1].access_count, reverse=True)
            all_facts_items = all_facts_items[:10]

        final_results = []

        for fact_str, edge in all_facts_items:
            stringified_items = [
                (str(key), str(value)) for key, value in edge.properties.items()
            ]
            sorted_items = sorted(stringified_items)

            final_results.append((fact_str, tuple(sorted_items)))

        return tuple(final_results)

    def _filter_facts_for_temporal_query(
        self,
        facts_with_props_tuple: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ) -> set[str]:
        """Filter a set of facts to find the most current one."""
        logger.debug("  [TemporalReasoning]: Filtering facts by date...")
        today = datetime.utcnow().date()
        best_fact: str | None = None
        best_date: date | None = None

        facts_list = [
            (fact_str, dict(props_tuple))
            for fact_str, props_tuple in facts_with_props_tuple
        ]

        for fact_str, props in facts_list:
            date_str = props.get("effective_date")
            if date_str:
                try:
                    fact_date = datetime.fromisoformat(date_str).date()
                    if fact_date <= today:
                        if best_date is None or fact_date > best_date:
                            best_date = fact_date
                            best_fact = fact_str
                except (ValueError, TypeError):
                    continue
        if best_fact:
            return {best_fact}
        return {
            fact_str
            for fact_str, props in facts_list
            if not props.get("effective_date")
        }

    def _sanitize_sentence_for_learning(self, sentence: str) -> str:
        """
        Pre-processes a raw sentence from any source.

        Makes it easier for thesymbolic parser to understand. This is the main gatekeeper for knowledge.
        """
        sanitized = re.sub(r"\s*\(.*?\)\s*", " ", sentence).strip()

        sanitized = re.sub(r"^.*?\)\s*", "", sanitized)

        sanitized = re.sub(
            r"^(In|According to)\s+[\w\s]+,\s*",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )

        sanitized = sanitized.split(";")[0]

        sanitized = re.sub(r"^\s*[:\-\–]+\s*", "", sanitized)

        return sanitized.strip()

    def _clean_phrase(self, phrase: str) -> str:
        """Clean and normalize a phrase for use as a concept or relation in the graph."""
        clean_phrase = phrase.lower().strip().replace("_", " ")

        clean_phrase = re.sub(r"\s*\([^)]*\)", "", clean_phrase).strip()

        clean_phrase = re.sub(r"[.,!?;']+$", "", clean_phrase)

        words = clean_phrase.split()
        if len(words) > 1 and words[0] in ("a", "an", "the"):
            return " ".join(words[1:]).strip()

        return clean_phrase

    def _extract_name_from_relation(
        self, value: str | dict[str, Any] | list | None
    ) -> str | None:
        """
        Extracts a clean name string from a relation's subject or object.
        This is hardened to handle the various data types returned by the LLM.
        """
        if value is None:
            return None

        if isinstance(value, list):
            value = " ".join(str(v) for v in value)

        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str):
                name_clean = name.strip()
                if name_clean:
                    return self._clean_phrase(name_clean)
            return None

        name_clean = value.strip()
        return self._clean_phrase(name_clean) or None

    def _process_statement_for_learning(
        self,
        relation: RelationData,
    ) -> tuple[bool, str]:
        """Process a structured fact, handling validation and belief revision."""
        if self.inference_mode:
            return False, "Agent is in read-only mode."

        subj_raw = cast("str | dict[str, Any] | list | None", relation.get("subject"))
        verb_raw = cast("str | None", relation.get("verb"))
        obj_raw = cast("str | dict[str, Any] | list | None", relation.get("object"))

        if subj_raw is None or verb_raw is None or obj_raw is None:
            return (
                False,
                "Incomplete fact structure: requires subject, verb, and object.",
            )

        subject_name = self._extract_name_from_relation(subj_raw)
        object_name = self._extract_name_from_relation(obj_raw)

        if subject_name is None:
            return False, "Could not determine the subject of the fact."
        if object_name is None:
            return False, "Could not determine the object of the fact."

        if subject_name == object_name or fuzz.ratio(subject_name, object_name) > 95:
            logger.info(
                "  [Learning Filter]: Skipping trivial fact (subject and object are too similar: '%s' vs '%s').",
                subject_name,
                object_name,
            )
            return (False, "trivial_fact")

        if len(object_name.split()) > 5:
            truncated = (
                (object_name[:60] + "...") if len(object_name) > 60 else object_name
            )
            msg = f"Deferred learning: The object '{truncated}' appears to be a description, not a concept."
            logger.warning(msg)
            return False, msg

        verb_cleaned = verb_raw.lower().strip()

        logger.info(
            "  [LEARNING]: Processing: %s -> %s -> %s",
            subject_name,
            verb_cleaned,
            object_name,
        )

        self.learning_iterations += 1

        sub_node = self._add_or_update_concept(subject_name)
        obj_node = self._add_or_update_concept(object_name)

        if not (sub_node and obj_node):
            return (
                False,
                f"Failed to create concepts for '{subject_name}' or '{object_name}'.",
            )

        relation_type = self.get_relation_type(verb_cleaned, subject_name, object_name)

        if relation_type in self._EXCLUSIVE_RELATIONS:
            was_learned, message = self._resolve_exclusive_conflict(
                sub_node,
                obj_node,
                relation_type,
                relation,
            )
        else:
            was_learned, message = self._add_new_fact(
                sub_node,
                obj_node,
                relation_type,
                relation,
            )

        if not was_learned:
            return False, message

        try:
            self._gather_facts_multihop.cache_clear()
        except Exception:
            logger.debug(
                "  [Cache]: could not clear reasoning cache (missing cache_clear).",
            )

        logger.info("  [Cache]: Cleared reasoning cache due to new knowledge.")
        self.save_brain()
        self.save_state()

        return True, "I understand. I have noted that."

    def _resolve_exclusive_conflict(
        self,
        sub_node: ConceptNode,
        obj_node: ConceptNode,
        relation_type: str,
        relation_data: RelationData,
    ) -> tuple[bool, str]:
        """Handle belief revision for relationships that must be unique."""
        conflict_edge = self.graph.find_exclusive_conflict(sub_node, relation_type)
        if not conflict_edge:
            return self._add_new_fact(sub_node, obj_node, relation_type, relation_data)

        rel_props = relation_data.get("properties") or {}
        candidate_confidence = float(rel_props.get("confidence", 0.95))
        candidate_provenance = rel_props.get("provenance", "user")
        candidate_rank = PROVENANCE_RANK.get(candidate_provenance, 0)
        candidate_strength = (candidate_confidence, candidate_rank)

        existing_confidence = getattr(conflict_edge, "weight", 0.6)
        existing_prov = conflict_edge.properties.get("provenance", "unknown")
        existing_rank = PROVENANCE_RANK.get(existing_prov, 0)
        existing_strength = (existing_confidence, existing_rank)

        if candidate_strength > existing_strength:
            logger.warning(
                "  [Belief Revision]: New fact is stronger. Deprecating old fact.",
            )
            self.graph.update_edge_properties(
                conflict_edge,
                {"superseded_by": obj_node.name},
            )
            self.graph.update_edge_weight(conflict_edge, 0.2)
            return self._add_new_fact(sub_node, obj_node, relation_type, relation_data)

        if candidate_strength < existing_strength:
            logger.warning(
                "  [Belief Revision]: Existing fact is stronger. Rejecting new fact.",
            )
            return (False, "existing_fact_stronger")

        logger.warning(
            "  [Belief Revision]: Stalemate detected. Triggering clarification.",
        )
        if self.harvester:
            target_node = self.graph.get_node_by_id(conflict_edge.target)
            if target_node:
                goal = (
                    f"RESOLVE_CONFLICT: {sub_node.name} --[{relation_type}]--> {obj_node.name} "
                    f"vs {target_node.name}"
                )
                if goal not in self.learning_goals:
                    self.learning_goals.append(goal)
                    self.save_state()
        return (False, "exclusive_conflict")

    def _add_new_fact(
        self,
        sub_node: ConceptNode,
        obj_node: ConceptNode,
        relation_type: str,
        relation_data: RelationData,
    ) -> tuple[bool, str]:
        """Add a new, non-conflicting fact to the knowledge graph."""
        edge_exists = any(
            edge.type == relation_type and edge.target == obj_node.id
            for edge in self.graph.get_edges_from_node(sub_node.id)
        )

        if edge_exists:
            logger.debug(
                "    - Fact already exists: %s --[%s]--> %s",
                sub_node.name,
                relation_type,
                obj_node.name,
            )
            return (False, "fact_exists")

        rel_props = relation_data.get("properties") or {}
        candidate = {
            "subject": sub_node.name,
            "verb": relation_type,
            "object": obj_node.name,
        }
        caller_name = f"{self.__class__.__name__}._add_new_fact"

        status = validate_and_add_relation(
            self,
            dict(candidate),
            rel_props,
            caller_name=caller_name,
        )

        if status in ("inserted", "replaced"):
            logger.info(
                "[success]Learned new fact: %s --[%s]--> %s (status=%s)[/success]",
                sub_node.name,
                relation_type,
                obj_node.name,
                status,
            )
            return (True, status)

        if status == "deferred":
            logger.warning(
                "    Deferred learning for %s --[%s]--> %s. (in %s)",
                sub_node.name,
                relation_type,
                obj_node.name,
                caller_name,
            )
        elif status == "contradiction_stored":
            logger.warning(
                "    Contradiction detected and stored for %s --[%s]--> %s.",
                sub_node.name,
                relation_type,
                obj_node.name,
            )

        return (False, status)

    def get_relation_type(self, verb: str, subject: str, object_: str) -> str:
        """Determine the semantic relationship type from a simple verb."""
        cleaned_verb = verb.replace("_", " ")

        if "agent" in subject.lower() and cleaned_verb in [
            "be",
            "is",
            "are",
            "is named",
        ]:
            if len(object_.split()) == 1 and object_[0].isupper():
                return "has name"

        return RELATION_TYPE_MAP.get(cleaned_verb, cleaned_verb)

    def learn_new_fact_autonomously(
        self, fact_sentence: str, source_topic: str | None = None
    ) -> bool:
        """Learns from a sentence by decomposing it into atomic facts and processing them."""
        caller_name = f"{self.__class__.__name__}.learn_new_fact_autonomously"
        logger.info(
            "[Autonomous Learning]: Attempting to learn fact: '%s'",
            fact_sentence,
        )

        atomic_relations = self.interpreter.decompose_sentence_to_relations(
            text=fact_sentence, main_topic=source_topic
        )

        if not atomic_relations:
            logger.warning(
                "  [Autonomous Learning]: Interpreter could not extract any atomic facts from the sentence. (in %s)",
                caller_name,
            )
            return False

        facts_learned_count = 0
        for relation_data in atomic_relations:
            props = relation_data.setdefault("properties", {})
            props.setdefault("confidence", 0.8)
            props.setdefault("provenance", "llm_decomposition")

            logger.info(
                "  [Autonomous Learning]: Interpreted Relation: %s", relation_data
            )

            success, _ = self._process_statement_for_learning(relation_data)
            if success:
                facts_learned_count += 1

        if facts_learned_count > 0:
            logger.info(
                "[Autonomous Learning]: Successfully learned and saved %d new fact(s).",
                facts_learned_count,
            )
            return True
        logger.warning(
            "[Autonomous Learning]: Failed to learn any new facts from the decomposed relations. (in %s)",
            caller_name,
        )
        return False

    def _add_or_update_concept(
        self,
        name: str,
        node_type: str = "concept",
    ) -> ConceptNode | None:
        clean_name = self._clean_phrase(name)
        if not clean_name:
            return None

        node = self.graph.get_node_by_name(clean_name)
        if not node:
            if " " in clean_name:
                determined_type = (
                    "proper_noun" if any(c.isupper() for c in name) else "noun_phrase"
                )
            else:
                word_info_list = get_word_info_from_wordnet(clean_name)
                if word_info_list and isinstance(word_info_list[0], dict):
                    determined_type = word_info_list[0].get("type", node_type)
                else:
                    determined_type = node_type

            node = self.graph.add_node(
                ConceptNode(clean_name, node_type=determined_type),
            )
            logger.info(
                "    Added new concept to graph: %s (%s)",
                clean_name,
                node.type,
            )
        return node

    def _add_or_update_concept_quietly(
        self,
        name: str,
        node_type: str = "concept",
    ) -> ConceptNode | None:
        """Find a concept node by name, creating it if it doesn't exist."""
        clean_name = self._clean_phrase(name)
        if not clean_name:
            return None
        node = self.graph.get_node_by_name(clean_name)
        if not node:
            if " " in clean_name:
                determined_type = (
                    "proper_noun" if any(c.isupper() for c in name) else "noun_phrase"
                )
            else:
                word_info_list = get_word_info_from_wordnet(clean_name)
                if word_info_list and isinstance(word_info_list[0], dict):
                    determined_type = word_info_list[0].get("type", node_type)
                else:
                    determined_type = node_type
            node = self.graph.add_node(
                ConceptNode(clean_name, node_type=determined_type),
            )
        return node

    def manual_add_knowledge_quietly(
        self,
        concept_name1: str,
        concept_type1: str,
        relation: str,
        concept_name2: str,
        weight: float = 0.5,
    ) -> None:
        """Add a structured fact directly to the knowledge graph WITHOUT logging."""
        node1 = self._add_or_update_concept_quietly(
            concept_name1,
            node_type=concept_type1,
        )
        node2 = self._add_or_update_concept_quietly(concept_name2)
        if node1 and node2:
            self.graph.add_edge(node1, node2, relation, weight)

    def save_brain(self) -> None:
        """Save the current knowledge graph to its JSON file."""
        if self.inference_mode:
            return

        lock_file = self.brain_file.parent / f"{self.brain_file.name}.lock"
        try:
            lock_file.touch()
            self.graph.save_to_file(self.brain_file)
            logger.info("Agent brain saved to %s", self.brain_file)
        finally:
            if lock_file.exists():
                lock_file.unlink()

    def save_state(self) -> None:
        """Save the agent's current operational state to its JSON file."""
        if not self.inference_mode:
            self._save_agent_state()
