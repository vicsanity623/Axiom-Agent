from __future__ import annotations

import logging
import re
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from thefuzz import process

from .graph_core import ConceptNode, RelationshipEdge

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent
    from .universal_interpreter import Entity, InterpretData, RelationData, Task


logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, agent: CognitiveAgent):
        """Initializes the ConversationManager with a reference to the main agent."""
        self.agent = agent

    def chat(self, user_input: str) -> str:
        """
        Process a single user input and return the agent's response by orchestrating
        a multi-stage cognitive pipeline.
        """
        logger.info(" \nUser: %s", user_input)
        self.agent.graph.decay_activations()

        if self.agent.is_awaiting_clarification:
            return self._handle_clarification(user_input)

        interpretations = self._get_interpretation(user_input)

        # --- FIX: Create direct, meaningful relationships instead of junk nodes ---
        if self.agent.user_node and interpretations:
            primary_interpretation = interpretations[0]
            # Instead of creating a junk 'interaction' node, we now create a
            # direct and meaningful 'talked_about' edge from the user to the topic.
            for topic in primary_interpretation.get("key_topics", []):
                # Ensure the topic is a valid concept before creating an edge
                topic_node = self.agent.knowledge_manager.get_or_create_concept(topic)
                if topic_node:
                    # This creates a direct link: (current user) --[talked_about]--> (topic)
                    self.agent.graph.add_edge(
                        self.agent.user_node, topic_node, "talked_about", weight=0.6
                    )
                    logger.info(f"   - Noted user interest in topic: '{topic}'")

        if not interpretations:
            reflex_response = self._handle_cognitive_reflex(user_input)
            if reflex_response:
                return reflex_response

            return "I'm sorry, I was unable to understand that."

        # The rest of your method is correct and unchanged.
        self._update_history(interpretations)
        structured_response = self._process_intent(interpretations[0], user_input)

        if len(interpretations) > 1:
            self._learn_from_extra_clauses(interpretations[1:])

        final_response, synthesizer_was_used = self._synthesize_response(
            structured_response, user_input
        )

        if synthesizer_was_used and not self.agent.inference_mode:
            self._learn_from_introspection(final_response, interpretations[0])

        return final_response

    def _get_corrected_entity(self, entity_name: str) -> str:
        """Uses fuzzy matching to correct potential typos in an entity name."""
        all_concepts = list(self.agent.graph.get_all_node_names())
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

    def _find_specific_fact(self, subject_name: str, relation_type: str) -> str | None:
        """Finds all facts of a specific type related to a subject and formats them."""
        subject_node = self.agent.graph.get_node_by_name(subject_name)
        if not subject_node:
            return None

        edges = self.agent.graph.get_edges_from_node(subject_node.id)
        relevant_edges = [edge for edge in edges if edge.type == relation_type]

        if not relevant_edges:
            return None

        object_names = []
        for edge in relevant_edges:
            target_node = self.agent.graph.get_node_by_id(edge.target)
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
        all_edges = self.agent.graph.get_all_edges()

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
            source_node = self.agent.graph.get_node_by_id(edge.source)
            target_node = self.agent.graph.get_node_by_id(edge.target)

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

    def _perform_multi_hop_query(
        self, start_node: ConceptNode, end_node: ConceptNode, max_hops: int = 3
    ) -> list[RelationshipEdge] | None:
        """Find a path of relationships between a start and end node."""
        queue: list[tuple[str, list[RelationshipEdge]]] = [(start_node.id, [])]
        visited: set[str] = {start_node.id}

        while queue:
            current_node_id, path = queue.pop(0)
            if len(path) >= max_hops:
                continue

            for edge in self.agent.graph.get_edges_from_node(current_node_id):
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
            source_node = self.agent.graph.get_node_by_id(edge.source)
            target_node = self.agent.graph.get_node_by_id(edge.target)
            if not source_node or not target_node:
                continue
            if i == 0:
                parts.append(
                    f"{source_node.name.capitalize()} {edge.type.replace('_', ' ')} {target_node.name}"
                )
            else:
                parts.append(
                    f"which in turn {edge.type.replace('_', ' ')} {target_node.name}"
                )
        return ", and ".join(parts) + "."

    def _answer_yes_no_question(self, relation: RelationData) -> str:
        """
        Answer a yes/no question by checking for direct facts, performing multi-hop
        inference for hierarchical relationships, and checking for contradictions.
        """
        subject_raw = relation.get("subject")
        object_raw = relation.get("object")
        if not isinstance(subject_raw, str) or not isinstance(object_raw, str):
            return "I'm not sure how to answer that."

        subject = self._get_corrected_entity(subject_raw)
        obj = self._get_corrected_entity(object_raw)

        subject_node = self.agent.graph.get_node_by_name(subject)
        if not subject_node:
            return f"I don't have any information about {subject}."

        # --- Multi-Hop Inference for 'is_a' relationships ---
        if relation.get("verb") == "is_a":
            queue: list[str] = [subject_node.id]
            visited: set[str] = {subject_node.id}
            path_found = False
            while queue:
                current_node_id = queue.pop(0)
                current_node = self.agent.graph.get_node_by_id(current_node_id)
                if current_node and current_node.name == obj:
                    path_found = True
                    break

                for edge in self.agent.graph.get_edges_from_node(current_node_id):
                    if edge.type == "is_a" and edge.target not in visited:
                        visited.add(edge.target)
                        queue.append(edge.target)

            if path_found:
                return f"Yes, based on what I know, {subject} is a type of {obj}."

        # Step 2: Direct positive match
        for edge in self.agent.graph.get_edges_from_node(subject_node.id):
            target_node = self.agent.graph.get_node_by_id(edge.target)
            if (
                target_node
                and target_node.name == obj
                and edge.type == relation.get("verb")
            ):
                return f"Yes, based on what I know, {subject} {edge.type.replace('_', ' ')} {obj}."

        # Step 3: Conflicting fact
        for edge in self.agent.graph.get_edges_from_node(subject_node.id):
            if edge.type in ["is_a", "has_property"]:
                target_node = self.agent.graph.get_node_by_id(edge.target)
                if target_node and target_node.name != obj:
                    return f"No, based on what I know, {subject} {edge.type.replace('_', ' ')} {target_node.name}, not {obj}."

        return f"I'm not sure. I don't have any information about whether {subject} has a relationship with {obj}."

    def is_concept_well_known(self, concept_name: str, threshold: int = 3) -> bool:
        """
        A more intelligent check to see if a concept is "well-known".
        """
        node = self.agent.graph.get_node_by_name(concept_name)
        if not node:
            return False
        edge_count = len(self.agent.graph.get_edges_from_node(node.id)) + len(
            self.agent.graph.get_edges_to_node(node.id)
        )
        return edge_count > threshold

    @lru_cache(maxsize=256)
    def _gather_facts_multihop(
        self, start_node_id: str, max_hops: int
    ) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
        """Gather all facts related to a starting node via graph traversal."""
        logger.info(
            "  [Cache]: MISS! Executing full multi-hop graph traversal for node ID: %s",
            start_node_id,
        )
        start_node = self.agent.graph.get_node_by_id(start_node_id)
        if not start_node:
            return ()
        found_facts: dict[str, RelationshipEdge] = {}
        queue: list[tuple[str, int]] = [(start_node_id, 0)]
        visited: set[str] = {start_node_id}
        while queue:
            current_node_id, current_hop = queue.pop(0)
            if current_hop >= max_hops:
                continue
            current_node = self.agent.graph.get_node_by_id(current_node_id)
            if not current_node:
                continue
            for edge in self.agent.graph.get_edges_from_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                target_node = self.agent.graph.get_node_by_id(edge.target)
                if target_node:
                    fact_str = f"{current_node.name} {edge.type.replace('_', ' ')} {target_node.name}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append((edge.target, current_hop + 1))
            for edge in self.agent.graph.get_edges_to_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                source_node = self.agent.graph.get_node_by_id(edge.source)
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
        query_text: str = "",
    ) -> set[str]:
        """
        Filter a set of facts based on temporal constraints in the query.
        Supports 'current' (default) and specific years (e.g., 'in 2010').
        """
        logger.debug("  [TemporalReasoning]: Filtering facts by date...")

        # Extract target year if present
        target_year = None
        match = re.search(r"\b(19|20)\d{2}\b", query_text)
        if match:
            target_year = int(match.group(0))
            logger.debug(f"  [TemporalReasoning]: Detected target year: {target_year}")

        today = datetime.utcnow().date()
        filtered_facts = set()

        facts_list = [
            (fact_str, dict(props_tuple))
            for fact_str, props_tuple in facts_with_props_tuple
        ]

        for fact_str, props in facts_list:
            date_str = props.get("effective_date")
            if not date_str:
                # If no date is present, we generally include it unless we are looking for a specific time
                # and strict filtering is desired. For now, include it as timeless truth.
                if not target_year:
                    filtered_facts.add(fact_str)
                continue

            try:
                # Handle YYYY format
                if len(date_str) == 4 and date_str.isdigit():
                    fact_year = int(date_str)
                    fact_date = date(fact_year, 1, 1)  # Default to Jan 1
                else:
                    fact_date = datetime.fromisoformat(date_str).date()
                    fact_year = fact_date.year

                if target_year:
                    # Check if the fact was active in the target year
                    # Ideally we need start/end dates. For now, we check if effective_date matches year
                    # or if it's a "start date" (implied) and no end date.
                    # This is a simplification.
                    if fact_year == target_year:
                        filtered_facts.add(fact_str)
                else:
                    # Default "current" logic: most recent fact <= today
                    # This logic in the original code was a bit specific to finding ONE best fact.
                    # Let's keep it simple: if it's in the past or today, it's valid history.
                    # But if the user asks "who IS...", we want the latest.
                    # The original logic tried to find the "best_date".
                    pass

            except (ValueError, TypeError):
                continue

        # If we have a target year, return matches
        if target_year:
            return (
                filtered_facts
                if filtered_facts
                else {f"I found no facts specifically for {target_year}."}
            )

        # Original "Current/Latest" Logic for non-specific queries
        best_fact: str | None = None
        best_date: date | None = None

        for fact_str, props in facts_list:
            date_str = props.get("effective_date")
            if date_str:
                try:
                    if len(date_str) == 4 and date_str.isdigit():
                        fact_date = date(int(date_str), 1, 1)
                    else:
                        fact_date = datetime.fromisoformat(date_str).date()

                    if fact_date <= today and (
                        best_date is None or fact_date > best_date
                    ):
                        best_date = fact_date
                        best_fact = fact_str
                except (ValueError, TypeError):
                    continue

        if best_fact:
            return {best_fact}

        # Fallback to timeless facts
        return {
            fact_str
            for fact_str, props in facts_list
            if not props.get("effective_date")
        }

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

    def _get_interpretation(self, user_input: str) -> list[InterpretData] | None:
        """
        Pipeline for parsing and interpreting user input, implementing a true
        "Symbolic First" strategy with a single return path for robustness.
        """
        interpretations: list[InterpretData] | None = None

        # --- Step 1: Try the SymbolicParser on the raw input FIRST. ---
        logger.debug(
            "  [Cognitive Flow]: Attempting fast-path symbolic parse on raw input..."
        )
        interpretations = self.agent.parser.parse(user_input)

        # --- Step 2: If the fast-path fails, run the full pipeline. ---
        if interpretations is None:
            logger.debug(
                "  [Cognitive Flow]: Fast-path failed. Running full preprocessing pipeline."
            )
            sanitized_input = (
                self.agent.knowledge_manager._sanitize_sentence_for_learning(user_input)
            )
            expanded_input = self.agent._expand_contractions(sanitized_input)
            contextual_input = self._resolve_references(expanded_input)
            normalized_input = self.agent._preprocess_self_reference(contextual_input)

            # --- Step 2a: Try the SymbolicParser AGAIN with the clean text. ---
            logger.debug(
                "  [Cognitive Flow]: Attempting deep-path symbolic parse on normalized text..."
            )
            interpretations = self.agent.parser.parse(normalized_input)

            # --- Step 2b: If all symbolic attempts fail, fall back to the LLM. ---
            if interpretations is None:
                logger.warning(
                    "  [Cognitive Flow]: All symbolic parsing failed. Falling back to LLM interpreter."
                )
                llm_interpretation = self.agent.interpreter.interpret(normalized_input)
                if llm_interpretation:
                    interpretations = [llm_interpretation]

        # --- Final Step: Log success and return the final result. ---
        if interpretations is not None:
            # This log is critical for confirming which path succeeded.
            provenance = (
                interpretations[0].get("provenance", "unknown")
                if interpretations
                else "no_interpretation"
            )
            if "symbolic" in provenance:
                logger.info("   - ✅ Understood via Symbolic Parser.")
            elif "llm" in provenance:
                logger.info("   - ✅ Understood via Universal Interpreter.")

        return interpretations

    def _handle_cognitive_reflex(self, normalized_input: str) -> str | None:
        """Handles cases where interpretation fails due to unknown words."""
        words = normalized_input.lower().split()
        unknown_words = [
            w for w in words if w and not self.agent.lexicon.is_known_word(w)
        ]
        if not unknown_words:
            return None
        word_to_learn = sorted(set(unknown_words))[0]
        new_task: Task = {"task": "INVESTIGATE", "topic": word_to_learn}
        if new_task not in self.agent.high_priority_tasks:
            self.agent.high_priority_tasks.insert(0, new_task)
            logger.info(
                "  [Cognitive Reflex]: Found unknown word '%s'. Queued for high-priority study.",
                word_to_learn,
            )
        return f"I'm not familiar with the word '{word_to_learn}'. I will add it to my study queue and research it later."

    def _update_history(self, interpretations: list[InterpretData]) -> None:
        """Updates the conversational history."""
        self.agent.structured_history.append(("user", interpretations))
        if len(self.agent.structured_history) > 10:
            self.agent.structured_history = self.agent.structured_history[-10:]

    def _learn_from_extra_clauses(
        self, extra_interpretations: list[InterpretData]
    ) -> None:
        """Processes additional factual clauses from a multi-clause input."""
        for extra in extra_interpretations:
            if extra.get("intent") == "statement_of_fact":
                if rel := extra.get("relation"):
                    self.agent.knowledge_manager._process_statement_for_learning(rel)

    def _learn_from_introspection(
        self, final_response: str, original_interpretation: InterpretData
    ) -> None:
        """
        Analyzes the agent's own synthesized response to learn new facts using
        the context-free symbolic parser.
        """
        # --- FIX: The entire block for calculating context_subject is removed ---
        # The new, more powerful parser does not require a context subject to be
        # passed to it, so we can simplify this method significantly.

        logger.info(
            "  [Introspection]: Analyzing synthesized response for new knowledge..."
        )

        new_interpretations = self.agent.parser.parse(final_response)

        if new_interpretations:
            self._learn_from_extra_clauses(new_interpretations)

    def _resolve_references(self, text: str) -> str:
        """
        Resolves simple pronouns by prioritizing the nearest valid noun that appears
        before the pronoun, ignoring common interrogative nouns.
        """
        pronouns_to_resolve = {"it", "they", "its", "their", "them"}
        words = text.split()
        pronoun_found_at_index: int | None = None
        for i, word in enumerate(words):
            if word.lower().strip(".,!?;") in pronouns_to_resolve:
                pronoun_found_at_index = i
                break

        if pronoun_found_at_index is None:
            return text

        antecedent: str | None = None

        interrogative_nouns = {"color", "name", "type", "kind", "sort", "way", "reason"}

        if pronoun_found_at_index > 0:
            words_before_pronoun = words[:pronoun_found_at_index]
            for word in reversed(words_before_pronoun):
                clean_word = word.strip(".,!?;")
                if (
                    self.agent.lexicon.is_part_of_speech(clean_word, "noun")
                    and clean_word not in interrogative_nouns
                ):
                    antecedent = clean_word
                    logger.debug(
                        "[Coreference]: Found local antecedent: '%s'", antecedent
                    )
                    break

        if not antecedent:
            logger.debug("[Coreference]: No local antecedent found. Checking history.")
            antecedent_raw: str | dict[str, Any] | list | None = None
            for speaker, interpretations in reversed(self.agent.structured_history):
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
            if isinstance(antecedent_raw, str):
                antecedent = antecedent_raw
            elif isinstance(antecedent_raw, dict):
                antecedent = antecedent_raw.get("name")

        if antecedent:
            clean_antecedent = self.agent._clean_phrase(antecedent)
            if not clean_antecedent:
                return text

            modified_text = re.sub(
                r"\b(it|they|them)\b",
                clean_antecedent,
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            modified_text = re.sub(
                r"\b(its|their)\b",
                f"{clean_antecedent}'s",
                modified_text,
                count=1,
                flags=re.IGNORECASE,
            )
            if modified_text != text:
                logger.info(
                    "  [Coreference]: Resolved pronouns, transforming '%s' to '%s'",
                    text,
                    modified_text,
                )
                return modified_text

        logger.debug("[Coreference]: No suitable antecedent found.")
        return text

    def _handle_clarification(self, user_input: str) -> str:
        """Handle the user's response after a contradiction was detected."""
        logger.info("  [Curiosity]: Processing user's clarification...")
        interpretation = self.agent.interpreter.interpret(user_input)
        entities = interpretation.get("entities", [])
        if not entities:
            logger.warning(
                "  [Curiosity]: No entities found in user's clarification. No update applied."
            )
            return "I'm sorry, I couldn't understand your clarification."
        entity_name_raw = entities[0].get("name")
        if not isinstance(entity_name_raw, str):
            return (
                "I'm sorry, I couldn't identify a clear concept in your clarification."
            )
        correct_answer_name = self.agent._clean_phrase(entity_name_raw)
        subject_name: str | None = self.agent.clarification_context.get("subject")
        relation_type = self.agent.clarification_context.get("conflicting_relation")
        if not subject_name or not relation_type:
            logger.warning(
                "  [Curiosity]: Clarification context incomplete (subject=%s, relation=%s).",
                subject_name,
                relation_type,
            )
            return "I'm sorry, I cannot process this clarification."
        subject_node = self.agent.graph.get_node_by_name(subject_name)
        if not subject_node:
            logger.warning(
                "  [Curiosity]: Subject node '%s' not found in graph. Creating it.",
                subject_name,
            )
            subject_node = self.agent.graph.add_node(ConceptNode(name=subject_name))
        correct_node = self.agent.graph.get_node_by_name(correct_answer_name)
        if not correct_node:
            logger.info(
                "  [Curiosity]: Creating node for correct answer '%s'.",
                correct_answer_name,
            )
            correct_node = self.agent.graph.add_node(
                ConceptNode(name=correct_answer_name)
            )

        self._gather_facts_multihop.cache_clear()

        logger.info(
            "[border][CleanCache]: Cleared reasoning cache due to knowledge correction.[/border]"
        )
        updated_any = False
        for u, v, key, data in list(
            self.agent.graph.graph.out_edges(subject_node.id, keys=True, data=True)
        ):
            if data.get("type") == relation_type:
                target_node = self.agent.graph.get_node_by_id(v)
                if (
                    target_node
                    and self.agent._clean_phrase(target_node.name)
                    == correct_answer_name
                ):
                    self.agent.graph.graph[u][v][key]["weight"] = 1.0
                    logger.info(
                        "    - REINFORCED: %s --[%s]--> %s",
                        subject_name,
                        relation_type,
                        correct_answer_name,
                    )
                    updated_any = True
                else:
                    self.agent.graph.graph[u][v][key]["weight"] = 0.1
                    if target_node:
                        logger.info(
                            "    - PUNISHED: %s --[%s]--> %s",
                            subject_name,
                            relation_type,
                            target_node.name,
                        )
                    updated_any = True
        if not any(
            (node := self.agent.graph.get_node_by_id(v))
            and self.agent._clean_phrase(node.name) == correct_answer_name
            and data.get("type") == relation_type
            for u, v, key, data in self.agent.graph.graph.out_edges(
                subject_node.id, keys=True, data=True
            )
        ):
            logger.info(
                "  [Curiosity]: Adding missing edge for clarification: %s --[%s]--> %s",
                subject_name,
                relation_type,
                correct_answer_name,
            )
            self.agent.graph.add_edge(
                subject_node, correct_node, relation_type, weight=1.0
            )
            updated_any = True
        if updated_any:
            self.agent.save_brain()
            self.agent.is_awaiting_clarification = False
            self.agent.clarification_context = {}
            return "Thank you for the clarification. I have updated my knowledge."
        logger.warning("  [Curiosity]: No updates were applied despite clarification.")
        return "I could not update my knowledge from your clarification."

    def _process_intent(self, interpretation: InterpretData, user_input: str) -> str:
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
            response: str | None = self.agent._answer_question_about(
                "agent", user_input
            )
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
            was_learned, learn_msg = (
                self.agent.knowledge_manager._process_statement_for_learning(relation)
            )
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
            subject_name = self.agent._extract_name_from_relation(subject_raw)
            verb = relation.get("verb")
            if subject_name and verb:
                corrected_subject = self._get_corrected_entity(subject_name)
                subject_node = self.agent.graph.get_node_by_name(corrected_subject)
                if subject_node:
                    for edge in self.agent.graph.get_edges_from_node(subject_node.id):
                        if edge.type == verb:
                            target_node = self.agent.graph.get_node_by_id(edge.target)
                            if target_node:
                                property_name = verb.replace("has_", "").replace(
                                    "_", " "
                                )
                                return f"The {property_name} of {subject_name.capitalize()} is {target_node.name.capitalize()}."
            return f"I don't have information about the {relation.get('verb', 'property').replace('has_', '')} of {subject_name or 'that'}."
        if intent in ("question_about_entity", "question_about_concept"):
            if entities:
                entity_name = entities[0]["name"]
                corrected_entity_name = self._get_corrected_entity(entity_name)
                response = self.agent._answer_question_about(
                    corrected_entity_name, user_input
                )
                return (
                    response
                    if response is not None
                    else f"I don't have any specific information about '{corrected_entity_name}' right now."
                )
        return "I'm not sure how to process that. Could you rephrase?"

    def _synthesize_response(
        self, structured_response: str, user_input: str
    ) -> tuple[str, bool]:
        """Convert a structured, internal response into natural language."""
        if structured_response.endswith((".", "!", "?")):
            logger.debug(
                "  [Synthesizer]: Skipping synthesis for already complete sentence."
            )
            return (structured_response, False)

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
                "  [Synthesizer]: Structured input detected → %s", structured_response
            )
            fluent_response = self.agent.interpreter.synthesize(
                structured_response, original_question=user_input
            )
            logger.debug("  [Synthesizer]: Output → %s", fluent_response)
            return (fluent_response, True)

        logger.debug("  [Structured Response]: %s", structured_response)
        fluent_response = self.agent.interpreter.synthesize(
            structured_response, original_question=user_input
        )
        logger.debug("  [Synthesized Response]: %s", fluent_response)
        return (fluent_response, True)
