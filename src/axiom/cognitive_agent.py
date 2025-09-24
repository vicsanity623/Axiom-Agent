from __future__ import annotations

# cognitive_agent.py
import json
import os
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict

from .dictionary_utils import get_word_info_from_wordnet
from .graph_core import ConceptGraph, ConceptNode, RelationshipEdge
from .knowledge_base import seed_core_vocabulary, seed_domain_knowledge
from .lexicon_manager import LexiconManager
from .symbolic_parser import SymbolicParser
from .universal_interpreter import (
    InterpretData,
    RelationData,
    UniversalInterpreter,
)

if TYPE_CHECKING:
    from .universal_interpreter import Entity

BRAIN_FOLDER: Final = Path("brain")
DEFAULT_BRAIN_FILE: Final = BRAIN_FOLDER / "my_agent_brain.json"
DEFAULT_STATE_FILE: Final = BRAIN_FOLDER / "my_agent_state.json"


class ClarificationContext(TypedDict):
    subject: NotRequired[str]
    conflicting_relation: NotRequired[str]


class CognitiveAgent:
    def __init__(
        self,
        brain_file: Path = DEFAULT_BRAIN_FILE,
        state_file: Path = DEFAULT_STATE_FILE,
        load_from_file: bool = True,
        brain_data: dict | None = None,
        cache_data: dict | None = None,
        inference_mode: bool = False,
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
        print("Initializing Cognitive Agent...")
        self.brain_file = brain_file
        self.state_file = state_file
        self.inference_mode = inference_mode

        if self.inference_mode:
            print("   - Running in INFERENCE-ONLY mode. Learning is disabled.")

        self.interpreter = UniversalInterpreter()
        self.lexicon = LexiconManager(self)
        self.parser = SymbolicParser(self)
        self.learning_goals: list[str] = []

        if load_from_file:
            print(f"   - Loading brain from file: {self.brain_file}")
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
                print(
                    "   - CRITICAL FAILURE: Agent's core identity is missing. Re-seeding brain for integrity.",
                )
                self.graph = ConceptGraph()
                seed_domain_knowledge(self)
                seed_core_vocabulary(self)
                self.save_brain()
                self.save_state()

        elif brain_data is not None and cache_data is not None:
            print("   - Initializing brain from loaded .axm model data.")
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
        self.conversation_history: list[str] = []
        self.enable_contextual_memory = False
        self.autonomous_cycle_count = 0
        self.INTERPRETER_REBOOT_THRESHOLD = 150

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
        print(f"\nUser: {user_input}")
        self.graph.decay_activations()

        if self.is_awaiting_clarification:
            return self._handle_clarification(user_input)

        normalized_input = self._preprocess_self_reference(user_input)

        interpretation: InterpretData | None = None

        symbolic_interpretation = self.parser.parse(normalized_input)

        if symbolic_interpretation:
            interpretation = symbolic_interpretation
            print(
                "  [Cognitive Flow]: Symbolic parsing succeeded. Skipping LLM interpreter.",
            )
        else:
            words = normalized_input.lower().split()
            unknown_words = []
            for word in words:
                clean_word = re.sub(r"[^\w\s]", "", word)
                if clean_word and not self.lexicon.is_known_word(clean_word):
                    unknown_words.append(clean_word)

            if unknown_words:
                word_to_learn = sorted(set(unknown_words))[0]
                goal = f"INVESTIGATE: {word_to_learn}"
                if goal not in self.learning_goals:
                    self.learning_goals.append(goal)
                    print(
                        f"  [Cognitive Reflex]: I don't know the word '{word_to_learn}'. Adding to learning goals.",
                    )
                return f"I'm not familiar with the term '{word_to_learn}'. I'll need to research it before I can understand your sentence."

            print(
                "  [Cognitive Flow]: Symbolic parsing failed. Falling back to LLM interpreter.",
            )
            if self.enable_contextual_memory:
                interpretation = self.interpreter.interpret_with_context(
                    user_input,
                    self.conversation_history,
                )
            else:
                interpretation = self.interpreter.interpret(normalized_input)

        print(
            f"  [Interpreter Output]: Intent='{interpretation.get('intent', 'N/A')}', "
            f"Entities={[e.get('name') for e in interpretation.get('entities', [])]}, "
            f"Relation={interpretation.get('relation')}",
        )

        intent = interpretation.get("intent", "unknown")
        entities: list[Entity] = interpretation.get("entities", [])
        relation: RelationData | None = interpretation.get("relation")

        structured_response = self._process_intent(
            intent,
            entities,
            relation,
            user_input,
        )
        final_response = self._synthesize_response(structured_response, user_input)

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Agent: {final_response}")

        return final_response

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
        print("  [Curiosity]: Processing user's clarification...")
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
                print("  [Cache]: Cleared reasoning cache due to knowledge correction.")
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
                            print(
                                f"    - REINFORCED: {subject_name} --[{relation_type}]--> {correct_answer_name}",
                            )
                        else:
                            self.graph.graph[u][v][key]["weight"] = 0.1
                            if target_node_data:
                                print(
                                    f"    - PUNISHED: {subject_name} --[{relation_type}]--> {target_node_data.get('name')}",
                                )
                self.save_brain()

        self.is_awaiting_clarification = False
        self.clarification_context = {}
        final_response = "Thank you for the clarification. I have updated my knowledge."
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Agent: {final_response}")
        return final_response

    def _process_intent(
        self,
        intent: str | None,
        entities: list[Entity],
        relation: RelationData | None,
        user_input: str,
    ) -> str:
        """Route the interpreted user input to the appropriate cognitive function.

        This method acts as a central switchboard, taking the structured
        output from an interpreter (either symbolic or LLM) and deciding
        which specialized method to call. It handles routing for greetings,
        learning new facts, answering questions, and processing commands.

        Args:
            intent: The classified intent of the user's message.
            entities: A list of named entities extracted from the message.
            relation: The structured factual relationship, if one was found.
            user_input: The original raw text from the user.

        Returns:
            A structured, non-natural language string representing the
            result of the cognitive process. This is then passed to the
            synthesizer.
        """
        if intent == "greeting":
            return "Hello User."
        if intent == "farewell":
            return "Goodbye User."
        if intent in ("gratitude", "acknowledgment"):
            return "You're welcome!"
        if intent == "positive_affirmation":
            return "I'm glad you think so!"

        if intent == "statement_of_fact" and relation:
            _, response_message = self._process_statement_for_learning(relation)
            return response_message

        if intent == "statement_of_correction" and relation:
            print("  [Correction]: Processing user's correction...")
            was_learned, response_message = self._process_statement_for_learning(
                relation,
            )
            if was_learned:
                return "Thank you. I have corrected my knowledge."
            return f"I understood the correction, but failed to learn the new fact. Reason: {response_message}"

        if intent == "command" and "show all facts" in user_input.lower():
            return self._get_all_facts_as_string()

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

        return "Despite having concepts in my knowledge base, I currently lack high-confidence facts to display."

    def _answer_question_about(self, entity_name: str, user_input: str) -> str:
        """Find and return known facts related to a specific entity.

        This is the core reasoning function. It takes a named entity, finds
        it in the knowledge graph, and performs a multi-hop traversal to
        gather all related, high-confidence facts.

        It also handles special cases, such as questions about the agent
        itself, and can filter facts based on temporal queries (e.g.,
        questions containing "now" or "currently").

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

        print(f"  [CognitiveAgent]: Starting reasoning for '{entity_name}'.")
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
            return f"I know the concept of {subject_node.name.capitalize()}, but I don't have any specific details for that query."

        return ". ".join(sorted(facts)) + "."

    def _synthesize_response(self, structured_response: str, user_input: str) -> str:
        """Convert a structured, internal response into natural language.

        This method acts as the final step before replying to the user.
        It checks if the structured response is a simple, pre-defined
        phrase (e.g., "Hello User."). If it is, the response is returned
        directly.

        Otherwise, it passes the structured fact(s) to the LLM-based
        synthesizer to be converted into a fluent, conversational sentence.

        Args:
            structured_response: An internal, non-natural language string
                containing the agent's raw answer.
            user_input: The user's original question, used for context.

        Returns:
            The final, natural language response to be sent to the user.
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
            return structured_response

        print(f"  [Structured Response]: {structured_response}")
        fluent_response = self.interpreter.synthesize(
            structured_response,
            original_question=user_input,
        )
        print(f"  [Synthesized Response]: {fluent_response}")
        return fluent_response

    def _reboot_interpreter(self) -> None:
        """Destroy and recreate the UniversalInterpreter to ensure stability.

        This prophylactic measure is called periodically to prevent potential
        long-term memory leaks or state corruption in the underlying C++
        library of the LLM. It preserves and restores the interpretation
        and synthesis caches to maintain performance.
        """
        print(
            "\n--- [SYSTEM HEALTH]: Prophylactically rebooting Universal Interpreter ---",
        )
        print("This is a preventative measure to ensure long-term stability.")

        old_interp_cache = self.interpreter.interpretation_cache
        old_synth_cache = self.interpreter.synthesis_cache

        del self.interpreter
        self.interpreter = UniversalInterpreter()

        self.interpreter.interpretation_cache = old_interp_cache
        self.interpreter.synthesis_cache = old_synth_cache

        print(
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
        print(
            f"  [System Health]: Autonomous cycles since last interpreter reboot: {self.autonomous_cycle_count}/{self.INTERPRETER_REBOOT_THRESHOLD}",
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
                with open(self.state_file) as f:
                    state_data = json.load(f)
                    self.learning_iterations = state_data.get("learning_iterations", 0)
                print(
                    f"Agent state loaded from {self.state_file} (Learning Iterations: {self.learning_iterations}).",
                )
            except Exception:
                self.learning_iterations = 0
        else:
            self.learning_iterations = 0

    def _save_agent_state(self) -> None:
        """Save the agent's current operational state to its state file.

        Writes metadata, such as the `learning_iterations` counter, to
        the `my_agent_state.json` file for persistence between sessions.
        """
        state_data = {"learning_iterations": self.learning_iterations}
        with open(self.state_file, "w") as f:
            json.dump(state_data, f, indent=4)

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
            print(f"  [Pre-processor]: Normalized input to '{processed_text}'")
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
        print(
            f"  [Cache]: MISS! Executing full multi-hop graph traversal for node ID: {start_node_id}",
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
        print("  [TemporalReasoning]: Filtering facts by date...")
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

        clean_phrase = re.sub(r"[.,!?;]+$", "", clean_phrase)

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

        print(
            f"  [AGENT LEARNING: Processing interpreted statement: {subject_name} -> {verb} -> {objects_to_process}]",
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
                            print(
                                "  [Curiosity]: CONTRADICTION DETECTED (Exclusive Relationship)!",
                            )
                            conflicting_facts_str = (
                                f"Fact 1: {sub_node.name} {edge.type.replace('_', ' ')} {existing_target_data.get('name')}. "
                                f"Fact 2: {sub_node.name} {relation_type.replace('_', ' ')} {object_name}."
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
                    print(
                        f"    Learned new fact: {sub_node.name} --[{relation_type}]--> {obj_node.name} with properties {properties}",
                    )
                    learned_at_least_one = True
                else:
                    print(
                        f"    - Fact already exists: {sub_node.name} --[{relation_type}]--> {obj_node.name}",
                    )

        if learned_at_least_one:
            self._gather_facts_multihop.cache_clear()
            print("  [Cache]: Cleared reasoning cache due to new knowledge.")
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
        relation_type_map = {
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
        return relation_type_map.get(verb, verb.replace(" ", "_"))

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
            print("[Autonomous Learning]: Skipped. Agent is in inference mode.")
            return False
        print(f"[Autonomous Learning]: Attempting to learn fact: '{fact_sentence}'")
        interpretation = self.interpreter.interpret(fact_sentence)
        relation = interpretation.get("relation")
        print(f"  [Autonomous Learning]: Interpreted Relation: {relation}")
        if interpretation.get("intent") == "statement_of_fact" and relation:
            was_learned, response_message = self._process_statement_for_learning(
                relation,
            )
            if was_learned:
                print("[Autonomous Learning]: Successfully learned and saved new fact.")
                return True
            print(
                f"[Autonomous Learning]: Failed to process fact. Reason: {response_message}",
            )
        else:
            print(
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
            print(f"    Added new concept to graph: {clean_name} ({node.type})")
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
            print(
                f"Manually added knowledge: {concept_name1} --[{relation}]--> {concept_name2}",
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
