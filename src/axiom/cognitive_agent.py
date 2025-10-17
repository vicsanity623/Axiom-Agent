"""The core module for the Axiom Cognitive Agent.

This module is responsible for:
- Managing the entire cognitive loop from understanding user input,
to learning and responding.
"""

import json
import logging
import os
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Final, NotRequired, TypedDict

from nltk.stem import WordNetLemmatizer

from axiom.dictionary_utils import get_word_info_from_wordnet
from axiom.graph_core import ConceptGraph, ConceptNode, RelationshipEdge
from axiom.knowledge_base import seed_core_vocabulary, seed_domain_knowledge
from axiom.lexicon_manager import LexiconManager
from axiom.symbolic_parser import SymbolicParser
from axiom.universal_interpreter import (
    Entity,
    InterpretData,
    RelationData,
    UniversalInterpreter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CognitiveAgent")

lemmatizer = WordNetLemmatizer()

BRAIN_FOLDER: Final = Path("brain")
DEFAULT_BRAIN_FILE: Final = BRAIN_FOLDER / "my_agent_brain.json"
DEFAULT_STATE_FILE: Final = BRAIN_FOLDER / "my_agent_state.json"


class ClarificationContext(TypedDict):
    """Hold contextual information needed for a clarification request."""

    subject: NotRequired[str]
    conflicting_relation: NotRequired[str]


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


class CognitiveAgent:
    """Orchestrate the primary cognitive functions of the Axiom Agent.

    This class integrates all other components to execute a symbolic-first thought process.
    """

    INTERPRETER_REBOOT_THRESHOLD: ClassVar[int] = 50

    def __init__(
        self,
        brain_file: Path = DEFAULT_BRAIN_FILE,
        state_file: Path = DEFAULT_STATE_FILE,
        load_from_file: bool = True,
        brain_data: dict | None = None,
        cache_data: dict | None = None,
        inference_mode: bool = False,
        enable_llm: bool = True,
    ) -> None:
        """Initialize a new instance of the CognitiveAgent.

        This constructor sets up the agent's core components, including its
        interpreters, knowledge graph, and state. It can be initialized
        in one of two ways:
        1. From disk (`load_from_file=True`): Loads the brain and state
           from the specified JSON files. Performs a health check and
           re-seeds the brain if core identity is missing.
        2. From data (`load_from_file=False`): Loads the brain and cache
           directly from dictionaries, typically from an unpacked .axm model.

        Args:
            brain_file: Path to the agent's main knowledge graph file.
            state_file: Path to the agent's state file (e.g., counters).
            load_from_file: If True, load from files. If False, requires
                brain_data and cache_data.
            brain_data: A dictionary representing the knowledge graph.
            cache_data: A dictionary representing the interpreter cache.
            inference_mode: If True, disables all learning and saving
                functionality.
        """
        logger.info("Initializing Cognitive Agent...")
        self.brain_file = brain_file
        self.state_file = state_file
        self.inference_mode = inference_mode

        if self.inference_mode:
            logger.info("   - Running in INFERENCE-ONLY mode. Learning is disabled.")

        self.interpreter = UniversalInterpreter(load_llm=enable_llm)
        self.lexicon = LexiconManager(self)
        self.parser = SymbolicParser(self)
        self.lemmatizer = WordNetLemmatizer()
        self.learning_goals: list[str] = []

        if load_from_file:
            logger.info("   - Loading brain from file: %s", self.brain_file)
            self.graph = ConceptGraph.load_from_file(self.brain_file)
            self._load_agent_state()

            agent_node = self.graph.get_node_by_name("agent")
            name_edge_exists = False
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
                    name_edge_exists = True

            if not name_edge_exists:
                logger.critical(
                    "   - CRITICAL FAILURE: Missing identity, Re-seeding brain for integrity.",
                )
                self.graph = ConceptGraph()
                seed_domain_knowledge(self)
                seed_core_vocabulary(self)
                self.save_brain()
                self.save_state()

        elif brain_data is not None and cache_data is not None:
            logger.info("   - Initializing brain from loaded .axm model data.")
            self.graph = ConceptGraph.load_from_dict(brain_data)
            self.interpreter.interpretation_cache = dict(
                cache_data.get("interpretations", []),
            )
            self.interpreter.synthesis_cache = dict(cache_data.get("synthesis", []))
            self.learning_iterations = brain_data.get("learning_iterations", 0)
        else:
            raise ValueError("Agent must be initialized with either files or data.")

        self.is_awaiting_clarification = False
        self.clarification_context: ClarificationContext = ClarificationContext({})
        self.structured_history: list[tuple[str, list[InterpretData]]] = []
        self.autonomous_cycle_count = 0

    def chat(self, user_input: str) -> str:
        """Process a single user input and return the agent's response.

        This method orchestrates the agent's entire thought process for a
        single conversational turn. It follows a symbolic-first cognitive
        flow:

        1.  **Symbolic Parse:** Attempts to understand the input using the
            deterministic `SymbolicParser`.
        2.  **Cognitive Reflex:** If parsing fails, it checks for unknown
            words. If found, it creates a learning goal and returns a
            message stating its lack of knowledge.
        3.  **LLM Fallback:** If all words are known but the sentence is too
            complex, it falls back to the `UniversalInterpreter` (LLM).
        4.  **Intent Processing:** The structured interpretation (from either
            parser) is processed to learn, reason, or answer.
        5.  **Response Synthesis:** A final, natural language response is
            generated and returned.

        Args:
            user_input: The raw text message from the user.

        Returns:
            A natural language string representing the agent's final response.
        """
        logger.info(" \nUser: %s", user_input)
        self.graph.decay_activations()

        if self.is_awaiting_clarification:
            return self._handle_clarification(user_input)

        expanded_input = self._expand_contractions(user_input)
        contextual_input = self._resolve_references(expanded_input)
        normalized_input = self._preprocess_self_reference(contextual_input)
        interpretations: list[InterpretData] | None = self.parser.parse(
            normalized_input,
        )

        if not interpretations:
            words = normalized_input.lower().split()
            unknown_words = [
                re.sub(r"[^\w\s]", "", w)
                for w in words
                if w and not self.lexicon.is_known_word(re.sub(r"[^\w\s]", "", w))
            ]
            if unknown_words:
                word_to_learn = sorted(set(unknown_words))[0]
                goal = f"INVESTIGATE: {word_to_learn}"
                if goal not in self.learning_goals:
                    self.learning_goals.append(goal)
                    logger.info(
                        "  [Cognitive Reflex]: Unknown word '%s', added to learning goals.",
                        word_to_learn,
                    )
                return f"New word '{word_to_learn}' discovered, I must research it before I can understand."

            logger.warning(
                "  [Cognitive Flow]: Symbolic parsing failed. Falling back to LLM interpreter.",
            )
            llm_interpretation = self.interpreter.interpret(normalized_input)
            if llm_interpretation:
                interpretations = [llm_interpretation]

        if not interpretations:
            return "I'm sorry, I was unable to understand that."

        self.structured_history.append(("user", interpretations))
        if len(self.structured_history) > 10:
            self.structured_history = self.structured_history[-10:]

        primary_interpretation = interpretations[0]
        logger.info(
            "  [Interpreter Output]: Intent='%s', Entities=%s, Relation=%s",
            primary_interpretation.get("intent", "N/A"),
            [e.get("name") for e in primary_interpretation.get("entities", [])],
            primary_interpretation.get("relation"),
        )
        intent = primary_interpretation.get("intent", "unknown")
        entities: list[Entity] = primary_interpretation.get("entities", [])
        relation: RelationData | None = primary_interpretation.get("relation")

        structured_response = self._process_intent(
            intent,
            entities,
            relation,
            user_input,
        )

        if len(interpretations) > 1:
            for extra in interpretations[1:]:
                if extra.get("intent") == "statement_of_fact":
                    rel = extra.get("relation")
                    if rel:
                        self._process_statement_for_learning(rel)

        final_response, synthesizer_was_used = self._synthesize_response(
            structured_response,
            user_input,
        )

        if synthesizer_was_used and not self.inference_mode:
            introspection_context = None
            if intent.startswith("question") and relation and relation.get("subject"):
                introspection_context = relation["subject"]
            elif intent.startswith("question") and entities:
                introspection_context = entities[0]["name"]

            if introspection_context:
                logger.info(
                    "  [Introspection]: Analyzing synthesized response for new knowledge...",
                )
                new_interpretations = self.parser.parse(
                    final_response,
                    context_subject=introspection_context,
                )
                if new_interpretations:
                    for interp in new_interpretations:
                        if interp.get("intent") == "statement_of_fact":
                            new_rel = interp.get("relation")
                            if new_rel:
                                self._process_statement_for_learning(new_rel)

        return final_response

    def _resolve_references(self, text: str) -> str:
        """Resolve simple pronouns using the stored interpretations from history."""
        pronouns_to_resolve = {"it", "they", "its", "their", "them"}
        if not any(
            re.search(rf"\b{pronoun}\b", text, re.IGNORECASE)
            for pronoun in pronouns_to_resolve
        ):
            logger.debug("[Coreference]: No pronouns found in text.")
            return text

        antecedent = None
        for speaker, interpretations in reversed(self.structured_history):
            if speaker == "user" and interpretations:
                primary_interpretation = interpretations[0]
                relation = primary_interpretation.get("relation")
                entities = primary_interpretation.get("entities")

                if relation and relation.get("subject"):
                    antecedent = relation["subject"]
                    break
                if entities:
                    antecedent = entities[0]["name"]
                    break

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
        """Handle the user's response after a contradiction was detected.

        This method is triggered when the agent is in a special state,
        `is_awaiting_clarification`, which occurs after it has identified
        conflicting facts and asked the user for help.

        It interprets the user's input to determine the correct fact,
        then reinforces the correct relationship in the knowledge graph
        while punishing the incorrect ones by reducing their weights.

        Args:
            user_input: The user's message, expected to be the correct
                answer to the clarification question.

        Returns:
            A confirmation message to the user.
        """
        logger.info("  [Curiosity]: Processing user's clarification...")
        interpretation = self.interpreter.interpret(user_input)
        entities = interpretation.get("entities", [])

        if entities:
            correct_answer_name = self._clean_phrase(entities[0]["name"])
            subject_name: str | None = self.clarification_context.get("subject")
            relation_type = self.clarification_context.get("conflicting_relation")
            subject_node: ConceptNode | None = None
            if subject_name is not None:
                subject_node = self.graph.get_node_by_name(subject_name)

            if subject_node and relation_type:
                self._gather_facts_multihop.cache_clear()
                logger.info(
                    "  [Cache]: Cleared reasoning cache due to knowledge correction.",
                )
                for u, v, key, data in list(
                    self.graph.graph.out_edges(subject_node.id, keys=True, data=True),
                ):
                    if data.get("type") == relation_type:
                        target_node_data = self.graph.graph.nodes.get(v)
                        if (
                            target_node_data
                            and target_node_data.get("name") == correct_answer_name
                        ):
                            self.graph.graph[u][v][key]["weight"] = 1.0
                            logger.info(
                                "    - REINFORCED: %s --[%s]--> %s",
                                subject_name,
                                relation_type,
                                correct_answer_name,
                            )
                        else:
                            self.graph.graph[u][v][key]["weight"] = 0.1
                            if target_node_data:
                                logger.info(
                                    "    - PUNISHED: %s --[%s]--> %s",
                                    subject_name,
                                    relation_type,
                                    target_node_data.get("name"),
                                )
                self.save_brain()

        self.is_awaiting_clarification = False
        self.clarification_context = {}
        return "Thank you for the clarification. I have updated my knowledge."

    def _process_intent(
        self,
        intent: str | None,
        entities: list[Entity],
        relation: RelationData | None,
        user_input: str,
    ) -> str:
        """Route the interpreted user input to the appropriate cognitive function."""
        if intent == "greeting":
            return "Hello User."
        if intent == "farewell":
            return "Goodbye User."
        if intent in ("gratitude", "acknowledgment"):
            return "You're welcome!"
        if intent == "positive_affirmation":
            return "I'm glad you think so!"

        if intent == "statement_of_fact" and relation:
            subject = relation.get("subject")
            predicate = (
                relation.get("predicate")
                or relation.get("verb")
                or relation.get("relation")
            )
            obj = relation.get("object")

            if not subject or not predicate or not obj:
                return "I understood this as a factual statement, but some elements were missing."

            was_learned, learn_msg = self._process_statement_for_learning(relation)
            logger.info(
                "  [Knowledge Acquisition]: Learned = '%s', msg = '%s'",
                was_learned,
                learn_msg,
            )

            if was_learned:
                return "I understand. I have noted that."
            return f"I tried to record that fact but something went wrong: {learn_msg}"

        if intent == "statement_of_correction" and relation:
            logger.info("  [Correction]: Processing user's correction...")
            was_learned, response_message = self._process_statement_for_learning(
                relation,
            )
            if was_learned:
                return "Thank you. I have corrected my knowledge."
            return f"I understood the correction, but failed to learn the new fact. Reason: {
                response_message
            }"

        if intent == "command" and "show all facts" in user_input.lower():
            return self._get_all_facts_as_string()

        if intent == "question_yes_no" and relation:
            return self._answer_yes_no_question(relation)

        if intent in ("question_about_entity", "question_about_concept") and relation:
            start_concept = relation.get("subject")
            end_concept = relation.get("object")

            if isinstance(start_concept, str) and isinstance(end_concept, str):
                start_node = self.graph.get_node_by_name(start_concept)
                end_node = self.graph.get_node_by_name(end_concept)

                if start_node and end_node:
                    logger.debug(
                        "  [Multi-Hop]: Querying for path between '%s' and '%s'.",
                        start_node.name,
                        end_node.name,
                    )
                    path = self._perform_multi_hop_query(start_node, end_node)
                    if path:
                        explanation = self._format_path_as_sentence(path)
                        return f"Based on what I know: {explanation}"
                    return f"I don't know of a direct relationship between {start_concept} and {end_concept}."

        if intent in ("question_about_entity", "question_about_concept"):
            entity_name = entities[0]["name"] if entities else user_input
            return self._answer_question_about(entity_name, user_input)

        return "I'm not sure how to process that. Could you rephrase?"

    def _get_all_facts_as_string(self) -> str:
        """Retrieve, filter, and format all facts from the knowledge graph.

        This function queries the entire knowledge graph to extract all
        relationships. It then filters out low-confidence facts (weight < 0.8)
        and sorts the remaining high-confidence facts from strongest to
        weakest.

        The final output is a single, formatted string ready for display
        to the user.

        Returns:
            A formatted string of all high-confidence facts, or a message
            indicating that the knowledge base is empty or lacks strong facts.
        """
        all_facts = []
        reconstructed_edges = []
        for u, v, data in self.graph.graph.edges(data=True):
            full_data = data.copy()
            full_data["source"] = u
            full_data["target"] = v
            reconstructed_edges.append(RelationshipEdge.from_dict(full_data))

        if not reconstructed_edges:
            return "My knowledge base is currently empty."

        sorted_edges = sorted(
            reconstructed_edges,
            key=lambda edge: edge.weight,
            reverse=True,
        )
        for edge in sorted_edges:
            if edge.type == "might_relate" or edge.weight < 0.8:
                continue

            source_node_data = self.graph.graph.nodes.get(edge.source)
            target_node_data = self.graph.graph.nodes.get(edge.target)

            if source_node_data and target_node_data:
                fact_string = (
                    f"- {source_node_data.get('name').capitalize()} "
                    f"--[{edge.type}]--> "
                    f"{target_node_data.get('name').capitalize()} "
                    f"(Weight: {edge.weight:.2f}, "
                    f"Salience: {edge.access_count})"
                )
                all_facts.append(fact_string)

        if all_facts:
            return (
                "Here are all the high-confidence facts I know (strongest first):\n\n"
                + "\n".join(all_facts)
            )

        return "I currently lack high-confidence facts to display."

    def _answer_question_about(self, entity_name: str, user_input: str) -> str:
        """Find and return known facts related to a specific entity.

        This is the core reasoning function. It takes a named entity, finds
        it in the knowledge graph, and performs a multi-hop traversal to
        gather all related, high-confidence facts.

        It can now also perform multi-hop symbolic reasoning for questions
        that ask about a relationship between two entities.

        Args:
            entity_name: The primary subject of the user's question.
            user_input: The original, full text of the user's question.

        Returns:
            A structured string summarizing the known facts, or a message
            indicating that no information is available.
        """
        clean_entity_name = self._clean_phrase(entity_name)

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
                    name_node_data = self.graph.graph.nodes.get(name_edge.target)
                    if name_node_data:
                        return f"My name is {name_node_data.get('name').capitalize()}."
                    return (
                        "I know I have a name, but I can't seem to recall it right now."
                    )
                return "I don't have a name yet."
            return "I don't seem to have a concept of myself right now."

        subject_node = self.graph.get_node_by_name(clean_entity_name)
        if not subject_node:
            return f"I don't have any information about {entity_name}."

        logger.info(
            "  [CognitiveAgent]: Starting single-hop reasoning for '%s',.",
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
        """Convert a structured, internal response into natural language.

        This is the final step before replying to the user.
        It bypasses synthesis for simple canned phrases and
        invokes the LLM synthesizer for any structured knowledge.
        """

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
        """Convert a path of edges into a human-readable sentence.

        Args:
            path: A list of RelationshipEdge objects representing a path.

        Returns:
            A string explaining the chain of reasoning.
        """
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
        subject = relation["subject"]
        object_ = relation["object"]

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
        """Destroy and recreate the UniversalInterpreter to ensure stability.

        This prophylactic measure is called periodically to prevent potential
        long-term memory leaks or state corruption in the underlying C++
        library of the LLM. It preserves and restores the interpretation
        and synthesis caches to maintain performance.
        """
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
        """Increment the autonomous cycle counter and trigger an interpreter reboot if needed.

        This method is called by the KnowledgeHarvester after every
        Study or Discovery cycle. It serves as a heartbeat to track the
        interpreter's operational lifetime and trigger a prophylactic
        reboot when a threshold is reached.
        """
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
        """Load the agent's operational state from its state file.

        Reads the `my_agent_state.json` file to restore metadata that is
        not part of the core knowledge graph, such as the total number of
        learning iterations.
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    state_data = json.load(f)
                    self.learning_iterations = state_data.get("learning_iterations", 0)
                logger.info(...)
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
        """Save the agent's current operational state to its state file.

        Writes metadata, such as the `learning_iterations` counter, to
        the `my_agent_state.json` file for persistence between sessions.
        """
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
        """Normalize user input to replace self-references with a canonical name.

        This function replaces pronouns and direct references (e.g., "you",
        "your name") with a consistent, third-person reference ("the agent",
        "the agent's name"). This simplifies the downstream parsing and
        reasoning logic by ensuring the agent is always referred to in
        the same way.

        Args:
            text: The raw input string from the user.

        Returns:
            The normalized string with self-references replaced.
        """
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
            r"\byour\b",
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
        """Gather all facts related to a starting node via graph traversal.

        This method performs a Breadth-First Search (BFS) starting from a
        given node to find all connected facts up to a specified depth.
        It traverses both outgoing and incoming relationships.

        The results are cached using `lru_cache` for performance. If more
        than 10 facts are found, it applies heuristics to filter for
        relevance and salience before returning.

        Args:
            start_node_id: The unique ID of the node to start the traversal from.
            max_hops: The maximum number of relationships (depth) to traverse.

        Returns:
            A tuple of fact tuples. Each inner tuple contains the formatted
            fact string and another tuple of its sorted properties.
        """
        logger.info(
            "  [Cache]: MISS! Executing full multi-hop graph traversal for node ID: %s",
            start_node_id,
        )
        start_node_data = self.graph.graph.nodes.get(start_node_id)
        if not start_node_data:
            return ()

        found_facts: dict[str, RelationshipEdge] = {}
        queue: list[tuple[str, int]] = [(start_node_id, 0)]
        visited: set[str] = {start_node_id}

        while queue:
            current_node_id, current_hop = queue.pop(0)
            if current_hop >= max_hops:
                continue

            current_node_data = self.graph.graph.nodes.get(current_node_id)
            if not current_node_data:
                continue

            for edge in self.graph.get_edges_from_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                target_node_data = self.graph.graph.nodes.get(edge.target)
                if target_node_data:
                    fact_str = f"{current_node_data.get('name').capitalize()} {edge.type.replace('_', ' ')} {target_node_data.get('name').capitalize()}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append((edge.target, current_hop + 1))

            for edge in self.graph.get_edges_to_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                source_node_data = self.graph.graph.nodes.get(edge.source)
                if source_node_data:
                    fact_str = f"{source_node_data.get('name').capitalize()} {edge.type.replace('_', ' ')} {current_node_data.get('name').capitalize()}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.source not in visited:
                        visited.add(edge.source)
                        queue.append((edge.source, current_hop + 1))

        all_facts_items = list(found_facts.items())

        if len(all_facts_items) > 10:
            original_subject = start_node_data.get("name", "").capitalize()
            relevance_filtered = [
                (f, e) for f, e in all_facts_items if f.startswith(original_subject)
            ]
            if relevance_filtered:
                all_facts_items = relevance_filtered

        if len(all_facts_items) > 10:
            all_facts_items.sort(key=lambda item: item[1].access_count, reverse=True)
            all_facts_items = all_facts_items[:10]

        return tuple(
            (fact_str, tuple(sorted(edge.properties.items())))
            for fact_str, edge in all_facts_items
        )

    def _filter_facts_for_temporal_query(
        self,
        facts_with_props_tuple: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ) -> set[str]:
        """Filter a set of facts to find the most current one.

        This function is used for temporal reasoning. It iterates through
        a list of facts, checks for an 'effective_date' property, and
        returns the single most recent fact that is not in the future.

        If no facts have a valid date, it returns all non-temporal facts.

        Args:
            facts_with_props_tuple: A tuple of facts gathered from the graph,
                where each fact includes its properties.

        Returns:
            A set containing the single most current fact, or a set of all
            non-temporal facts if no dates were found.
        """
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

    def _clean_phrase(self, phrase: str) -> str:
        """Clean and normalize a phrase for use as a concept in the graph.

        - Converts to lowercase.
        - Removes leading articles ('a', 'an', 'the').
        - Removes common punctuation from the end of the phrase.

        Args:
            phrase: The raw string phrase to be cleaned.

        Returns:
            The normalized phrase.
        """
        clean_phrase = phrase.lower().strip()
        clean_phrase = re.sub(r"[.,!?;']+$", "", clean_phrase)

        words = clean_phrase.split()

        if words and words[0] in ["a", "an", "the"]:
            words = words[1:]

        return " ".join(words).strip()

    def _process_statement_for_learning(
        self,
        relation: RelationData,
    ) -> tuple[bool, str]:
        """Process a structured fact to learn and integrate it into the graph.

        This is the primary method for adding new knowledge. It takes a
        structured `RelationData` object, cleans the subject and object,
        determines the semantic relationship type, and adds the new nodes
        and edge to the knowledge graph.

        It also contains the core "curiosity" logic: if a new fact
        contradicts an existing exclusive relationship (e.g., a person
        can only have one name), it will trigger the clarification state.

        Args:
            relation: A TypedDict containing the subject, verb, and object
                of the fact to be learned.

        Returns:
            A tuple containing a boolean indicating if a new fact was
            learned, and a string message for the user.
        """
        if self.inference_mode:
            return (
                False,
                "I am currently in a read-only mode and cannot learn new facts.",
            )

        subject = relation.get("subject")
        verb = relation.get("verb")
        object_ = relation.get("object")
        properties = relation.get("properties")
        if not all([subject, verb, object_]):
            return (False, "I couldn't understand the structure of that fact.")

        subject_name_raw = subject.get("name") if isinstance(subject, dict) else subject
        if not subject_name_raw:
            return (False, "Could not determine the subject of the fact.")
        subject_name = subject_name_raw

        objects_to_process = []
        if isinstance(object_, list):
            for item in object_:
                if isinstance(item, dict):
                    name = item.get("entity") or item.get("name")
                else:
                    name = item

                if name:
                    objects_to_process.append(name)
        else:
            name = object_.get("name") if isinstance(object_, dict) else object_
            if name:
                objects_to_process.append(name)

        if not objects_to_process:
            return (False, "Could not determine the object(s) of the fact.")

        logger.info(
            "  [LEARNING: Processing statement: %s -> %s -> %s]",
            subject_name,
            verb,
            objects_to_process,
        )
        self.learning_iterations += 1

        assert verb is not None
        verb_cleaned = verb.lower().strip()
        sub_node = self._add_or_update_concept(subject_name)

        for object_name in objects_to_process:
            relation_type = self._get_relation_type(
                verb_cleaned,
                subject_name,
                object_name,
            )
            exclusive_relations = ["has_name", "is_capital_of", "is_located_in"]
            if relation_type in exclusive_relations and sub_node:
                for edge in self.graph.get_edges_from_node(sub_node.id):
                    if edge.type == relation_type:
                        existing_target_data = self.graph.graph.nodes[edge.target]
                        if existing_target_data.get("name") != self._clean_phrase(
                            object_name,
                        ):
                            logger.warning(
                                "  [Curiosity]: CONTRADICTION DETECTED (Exclusive Relationship)!",
                            )
                            conflicting_facts_str = (
                                f"Fact 1: {sub_node.name} {
                                    edge.type.replace('_', ' ')
                                } {existing_target_data.get('name')}. "
                                f"Fact 2: {sub_node.name} {
                                    relation_type.replace('_', ' ')
                                } {object_name}."
                            )
                            question = self.interpreter.synthesize(
                                conflicting_facts_str,
                                mode="clarification_question",
                            )
                            self.is_awaiting_clarification = True
                            self.clarification_context = {
                                "subject": sub_node.name,
                                "conflicting_relation": relation_type,
                            }
                            return (False, question)

        learned_at_least_one = False
        for object_name in objects_to_process:
            relation_type = self._get_relation_type(
                verb_cleaned,
                subject_name,
                object_name,
            )
            obj_node = self._add_or_update_concept(object_name)
            if sub_node and obj_node:
                edge_exists = any(
                    edge.type == relation_type and edge.target == obj_node.id
                    for edge in self.graph.get_edges_from_node(sub_node.id)
                )

                if not edge_exists:
                    self.graph.add_edge(
                        sub_node,
                        obj_node,
                        relation_type,
                        0.9,
                        properties=properties,
                    )
                    logger.info(
                        "    Learned new fact: %s --[%s]--> %s with properties %s",
                        sub_node.name,
                        obj_node.name,
                        relation_type,
                        properties,
                    )

                    learned_at_least_one = True
                else:
                    logger.debug(
                        "    - Fact already exists: %s --[%s]--> %s",
                        sub_node.name,
                        relation_type,
                        obj_node.name,
                    )

        if learned_at_least_one:
            self._gather_facts_multihop.cache_clear()
            logger.info("  [Cache]: Cleared reasoning cache due to new knowledge.")
            self.save_brain()
            self.save_state()
            return (True, "I understand. I have noted that.")

        return (True, "I have processed that information.")

    def _get_relation_type(self, verb: str, subject: str, object_: str) -> str:
        """Determine the semantic relationship type from a simple verb.

        This function converts a simple verb (like "is" or "locate in")
        into a more structured, semantic relationship type for the graph
        (e.g., "is_a", "is_located_in"). It includes special-case logic,
        such as identifying the "has_name" relationship.

        If no specific mapping is found, it defaults to using the verb
        itself, formatted as snake_case.

        Args:
            verb: The verb from the parsed relation.
            subject: The subject of the relation.
            object_: The object of the relation.

        Returns:
            The formatted, semantic relationship type as a string.
        """
        if "agent" in subject.lower() and verb in [
            "be",
            "is",
            "are",
            "is named",
            "is_named",
        ]:
            if len(object_.split()) == 1 and object_[0].isupper():
                return "has_name"
        return RELATION_TYPE_MAP.get(verb, verb.replace(" ", "_"))

    def learn_new_fact_autonomously(self, fact_sentence: str) -> bool:
        """Process and learn a fact from a raw sentence string.

        This is a high-level wrapper used by the KnowledgeHarvester. It
        takes a raw sentence, sends it to the LLM interpreter, and if a
        valid factual statement is returned, it passes the structured
        relation to the main learning processor.

        Args:
            fact_sentence: A string containing a potential fact to learn.

        Returns:
            True if a new fact was successfully learned, False otherwise.
        """
        if self.inference_mode:
            logger.info("[Autonomous Learning]: Skipped. Agent is in inference mode.")
            return False
        logger.info(
            "[Autonomous Learning]: Attempting to learn fact: '%s'",
            fact_sentence,
        )
        interpretation = self.interpreter.interpret(fact_sentence)
        relation = interpretation.get("relation")
        logger.info(
            "  [Autonomous Learning]: Interpreted Relation: %s",
            relation,
        )
        if interpretation.get("intent") == "statement_of_fact" and relation:
            was_learned, response_message = self._process_statement_for_learning(
                relation,
            )
            if was_learned:
                logger.info(
                    "[Autonomous Learning]: Successfully learned and saved new fact.",
                )
                return True
            logger.warning(
                "[Autonomous Learning]: Failed to process fact. Reason: %s",
                response_message,
            )
        else:
            logger.warning(
                "[Autonomous Learning]: Could not interpret the sentence as a statement of fact.",
            )
        return False

    def _add_or_update_concept(
        self,
        name: str,
        node_type: str = "concept",
    ) -> ConceptNode | None:
        """Find a concept node by name, creating it if it doesn't exist.

        This is a core utility for managing concepts. It cleans the given
        name, checks if a node with that name already exists, and returns
        it. If not, it creates a new `ConceptNode`, attempting to infer a
        more specific type (e.g., 'proper_noun') before adding it to the
        graph.

        Args:
            name: The name of the concept to find or create.
            node_type: The default node type to assign if creating a new node.

        Returns:
            The found or newly created ConceptNode, or None if the name is invalid.
        """
        clean_name = self._clean_phrase(name)
        if not clean_name:
            return None
        node = self.graph.get_node_by_name(clean_name)
        if not node:
            if len(clean_name.split()) > 1:
                determined_type = (
                    "proper_noun" if any(c.isupper() for c in name) else "noun_phrase"
                )
            else:
                determined_type = get_word_info_from_wordnet(clean_name).get(
                    "type",
                    node_type,
                )
            node = self.graph.add_node(
                ConceptNode(clean_name, node_type=determined_type),
            )
            logger.info(
                "    Added new concept to graph: %s (%s)",
                clean_name,
                node.type,
            )
        return node

    def manual_add_knowledge(
        self,
        concept_name1: str,
        concept_type1: str,
        relation: str,
        concept_name2: str,
        weight: float = 0.5,
    ) -> None:
        """Add a structured fact directly to the knowledge graph.

        This is a helper method used primarily for seeding the brain with
        initial knowledge. It bypasses all interpreters and directly
        creates the specified nodes and relationship.

        Args:
            concept_name1: The name of the source concept.
            concept_type1: The type of the source concept.
            relation: The relationship type for the edge.
            concept_name2: The name of the target concept.
            weight: The confidence weight for the new fact.
        """
        node1 = self._add_or_update_concept(concept_name1, node_type=concept_type1)
        node2 = self._add_or_update_concept(concept_name2)
        if node1 and node2:
            self.graph.add_edge(node1, node2, relation, weight)
            logger.info(
                "Manually added knowledge: %s --[%s]--> %s",
                concept_name1,
                relation,
                concept_name2,
            )

    def save_brain(self) -> None:
        """Save the current knowledge graph to its JSON file.

        This method persists the agent's long-term memory. It will not
        execute if the agent is in inference-only mode.
        """
        if not self.inference_mode:
            self.graph.save_to_file(self.brain_file)

    def save_state(self) -> None:
        """Save the agent's current operational state to its JSON file.

        This method persists metadata about the agent's lifetime, such as
        learning counters. It is a convenience wrapper around the private
        `_save_agent_state` method.
        """
        if not self.inference_mode:
            self._save_agent_state()
