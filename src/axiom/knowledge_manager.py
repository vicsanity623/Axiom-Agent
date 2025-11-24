from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, cast

from nltk.stem import WordNetLemmatizer

from .constants import PROVENANCE_RANK
from .dictionary_utils import get_word_info_from_wordnet
from .graph_core import ConceptNode
from .knowledge_base import add_pending_relation, validate_and_add_relation
from .universal_interpreter import PropertyData, RelationData, Task

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

POS_CONCEPTS = frozenset(
    [
        "noun",
        "verb",
        "adjective",
        "descriptor",
        "adverb",
        "pronoun",
        "preposition",
        "conjunction",
        "article",
        "determiner",
        "interjection",
    ]
)


class KnowledgeManager:
    def __init__(self, agent: CognitiveAgent):
        """Initializes the KnowledgeManager with a reference to the main agent."""
        self.agent = agent

    def get_or_create_concept(
        self,
        name: str,
        node_type: str = "concept",
        context_relation_type: str | None = None,
        context_object_name: str | None = None,
    ) -> ConceptNode | None:
        """
        Public API to find a concept by name or create it. This method handles
        the full lifecycle, including homonym disambiguation.
        """
        return self._get_or_create_concept_node(
            name, node_type, context_relation_type, context_object_name
        )

    def _get_or_create_concept_node(
        self,
        name: str,
        node_type: str = "concept",
        context_relation_type: str | None = None,
        context_object_name: str | None = None,
        context_sentence: str | None = None,
    ) -> ConceptNode | None:
        """
        Internal implementation for finding or creating a concept node, with
        corrected logic to handle homonyms and ensure all new nodes have
        rich default properties.
        """
        clean_name = self.agent._clean_phrase(name)
        if not clean_name:
            return None

        existing_node = self.agent.graph.get_node_by_name(clean_name)

        # --- FIX: Change the return type hint and use cast ---
        def create_default_properties(determined_type: str) -> PropertyData:
            lexical_info = {}
            if determined_type not in (
                "concept",
                "noun_phrase",
                "self_identity",
                "user_identity",
            ):
                lexical_info = {determined_type: 0.9}

            return cast(
                "PropertyData",
                {
                    "created_at": datetime.utcnow().isoformat(),
                    "last_accessed": datetime.utcnow().isoformat(),
                    "access_count": 1,
                    "lexical_promoted_as": lexical_info,
                },
            )

        # Path 1: The word is completely new. Create it and return.
        if not existing_node:
            if " " in clean_name:
                determined_type = "noun_phrase"
            else:
                word_info_list = get_word_info_from_wordnet(clean_name)
                determined_type = (
                    word_info_list[0].get("type", node_type)
                    if word_info_list and isinstance(word_info_list[0], dict)
                    else node_type
                )

            default_props = create_default_properties(determined_type)
            new_node_obj = ConceptNode(
                clean_name, node_type=determined_type, properties=default_props
            )

            node = self.agent.graph.add_node(new_node_obj)
            logger.info(
                f"[cyan]   - Added new concept to graph: {clean_name} ({node.type})[/cyan]"
            )
            return node

        # Path 2: Check for existing senses (disambiguation)
        # If we have a context sentence, we attempt disambiguation to see if we should map to a specific sense.
        # This allows us to start splitting a concept into senses even if none exist yet.
        if context_sentence:
            sense_label = self.agent.interpreter.disambiguate_word(
                clean_name, context_sentence
            )
            if sense_label:
                specific_sense_name = f"{clean_name} ({sense_label})"
                existing_sense = self.agent.graph.get_node_by_name(specific_sense_name)
                if existing_sense:
                    logger.info(
                        f"   - Resolved '{clean_name}' to existing sense: '{specific_sense_name}'"
                    )
                    return existing_sense

                # Create new sense node
                logger.info(
                    f"   - Creating new sense for '{clean_name}': '{specific_sense_name}'"
                )
                determined_type = existing_node.type  # Inherit type usually
                default_props = create_default_properties(determined_type)
                new_sense_node = ConceptNode(
                    specific_sense_name,
                    node_type=determined_type,
                    properties=default_props,
                )
                self.agent.graph.add_node(new_sense_node)
                self.agent.graph.add_edge(existing_node, new_sense_node, "has_sense")
                return new_sense_node

        # Path 3: The word exists, but no context is provided OR disambiguation failed. Use the existing node.
        if not context_relation_type or not context_object_name:
            return existing_node

        conflicting_edges = [
            edge
            for edge in self.agent.graph.get_edges_from_node(existing_node.id)
            if edge.type == context_relation_type
        ]

        # Path 4: No relationships of this type exist yet. Use the base node.
        if not conflicting_edges:
            return existing_node

        # Path 5: Check if a relationship already points to our target object.
        for edge in conflicting_edges:
            target_node = self.agent.graph.get_node_by_id(edge.target)
            if target_node and target_node.name == context_object_name:
                return existing_node

        # Path 6: A true ambiguity is detected via conflicting facts. Create a new "sense" node.
        # This is the fallback if LLM disambiguation wasn't triggered but we see a conflict.
        new_sense_name = f"{clean_name} ({context_object_name})"
        logger.info(
            f"   - Ambiguity detected for '{clean_name}'. Creating specific sense: '{new_sense_name}'"
        )

        existing_sense_node = self.agent.graph.get_node_by_name(new_sense_name)
        if existing_sense_node:
            return existing_sense_node

        if " " in clean_name:
            determined_type = "noun_phrase"
        else:
            word_info_list = get_word_info_from_wordnet(clean_name)
            determined_type = (
                word_info_list[0].get("type", node_type)
                if word_info_list and isinstance(word_info_list[0], dict)
                else node_type
            )

        default_props = create_default_properties(determined_type)
        new_node_obj = ConceptNode(
            new_sense_name, node_type=determined_type, properties=default_props
        )

        node = self.agent.graph.add_node(new_node_obj)
        # Link base to sense
        self.agent.graph.add_edge(existing_node, node, "has_sense")

        logger.info(
            f"[cyan]   - Added new concept to graph: {new_sense_name} ({node.type})[/cyan]"
        )
        return node

    def _process_statement_for_learning(
        self,
        relation: RelationData,
        allow_unknowns_for_topic: str | None = None,
    ) -> tuple[bool, str]:
        """
        Process a structured fact, performing validation BEFORE node creation.
        """
        if self.agent.inference_mode:
            return False, "Agent is in read-only mode."

        subject_name = self.agent._extract_name_from_relation(relation.get("subject"))
        object_name = self.agent._extract_name_from_relation(relation.get("object"))
        verb_name = self.agent._extract_name_from_relation(relation.get("verb"))

        if not subject_name or not object_name or not verb_name:
            return False, "Incomplete fact structure."

        # After the check above, mypy knows these are all strings.
        if subject_name == object_name:
            return False, "trivial_fact"

        all_words = set(subject_name.split()) | set(object_name.split())
        unknown_words = [
            word for word in all_words if not self.agent.lexicon.is_known_word(word)
        ]

        if unknown_words:
            props = relation.get("properties", {}) or {}
            is_defining_new_word = props.get("provenance") in (
                "dictionary_api",
                "dictionary_pos",
            )

            final_unknowns = []
            for word in unknown_words:
                if is_defining_new_word and word == subject_name:
                    continue
                if allow_unknowns_for_topic and word == allow_unknowns_for_topic:
                    continue
                final_unknowns.append(word)

            if final_unknowns:
                logger.warning(
                    f"[KnowledgeManager] Deferring fact due to unknown words: {final_unknowns}"
                )
                clean_unknown_words = {
                    self.agent._clean_phrase(word)
                    for word in final_unknowns
                    if self.agent._clean_phrase(word)
                }

                for clean_word in clean_unknown_words:
                    goal = Task(
                        task="INVESTIGATE",
                        topic=clean_word,
                        context=f"related to the concept of '{subject_name}'",
                    )
                    if goal not in self.agent.high_priority_tasks:
                        self.agent.high_priority_tasks.insert(0, goal)

                add_pending_relation(
                    self.agent, relation, props, list(clean_unknown_words)
                )
                return False, "deferred"

        logger.info(
            f"[cyan]   - Processing: {subject_name} -> {verb_name} -> {object_name}[/cyan]"
        )
        relation_type = self.agent.get_relation_type(
            verb_name, subject_name, object_name
        )

        obj_node = self.get_or_create_concept(object_name)

        # Reconstruct context sentence if possible, or use what we have
        # relation.get("properties", {}).get("full_text_rephrased") might be available if we passed it down
        # For now, we can reconstruct a simple one:
        context_sentence = f"{subject_name} {verb_name} {object_name}"

        sub_node = self._get_or_create_concept_node(
            subject_name,
            context_relation_type=relation_type,
            context_object_name=object_name,
            context_sentence=context_sentence,
        )

        if not sub_node or not obj_node:
            return False, "Failed to create nodes for known concepts."

        if relation_type in self.agent._EXCLUSIVE_RELATIONS:
            was_learned, message = self._resolve_exclusive_conflict(
                sub_node, obj_node, relation_type, relation, allow_unknowns_for_topic
            )
        else:
            was_learned, message = self._add_new_fact(
                sub_node, obj_node, relation_type, relation, allow_unknowns_for_topic
            )

        if was_learned:
            try:
                self.agent.conversation_manager._gather_facts_multihop.cache_clear()
                logger.info(
                    "[border]   - Cleared reasoning cache due to new knowledge.[/border]"
                )
            except Exception:
                logger.debug("  [Cache]: could not clear reasoning cache.")
            self.agent.save_brain()
            self.agent.save_state()
            return True, "I understand. I have noted that."

        return False, message

    def _resolve_exclusive_conflict(
        self,
        sub_node: ConceptNode,
        obj_node: ConceptNode,
        relation_type: str,
        relation_data: RelationData,
        allow_unknowns_for_topic: str | None = None,
    ) -> tuple[bool, str]:
        """Handle belief revision for relationships that must be unique."""
        conflict_edge = self.agent.graph.find_exclusive_conflict(
            sub_node, relation_type
        )
        if not conflict_edge:
            return self._add_new_fact(
                sub_node,
                obj_node,
                relation_type,
                relation_data,
                allow_unknowns_for_topic,
            )

        rel_props = relation_data.get("properties") or {}
        candidate_confidence = float(rel_props.get("confidence", 0.95))
        candidate_provenance = str(rel_props.get("provenance", "user"))
        candidate_rank = PROVENANCE_RANK.get(candidate_provenance, 0)
        candidate_strength = (candidate_confidence, candidate_rank)

        existing_confidence = getattr(conflict_edge, "weight", 0.6)
        existing_prov = str(conflict_edge.properties.get("provenance", "unknown"))
        existing_rank = PROVENANCE_RANK.get(existing_prov, 0)
        existing_strength = (existing_confidence, existing_rank)

        if candidate_strength > existing_strength:
            self.agent.graph.update_edge_properties(
                conflict_edge,
                cast("dict[str, object]", {"revision_status": "superseded"}),
            )
            return self._add_new_fact(
                sub_node,
                obj_node,
                relation_type,
                relation_data,
                allow_unknowns_for_topic,
            )

        if candidate_strength < existing_strength:
            return False, "existing_fact_stronger"

        return False, "exclusive_conflict"

    def _sanitize_sentence_for_learning(self, sentence: str) -> str:
        """
        Pre-processes a raw sentence from any source.
        """
        sanitized = re.sub(r"\s*\(.*?\)\s*", " ", sentence).strip()
        sanitized = re.sub(r"^.*?\)\s*", "", sanitized)
        sanitized = re.sub(
            r"^(In|According to)\s+[\w\s]+,\s*", "", sanitized, flags=re.IGNORECASE
        )
        sanitized = sanitized.split(";")[0]
        sanitized = re.sub(r"^\s*[:\-\–]+\s*", "", sanitized)
        return sanitized.strip()

    def _add_new_fact(
        self,
        sub_node: ConceptNode,
        obj_node: ConceptNode,
        relation_type: str,
        relation_data: RelationData,
        allow_unknowns_for_topic: str | None = None,
    ) -> tuple[bool, str]:
        """
        A simple wrapper that delegates fact addition to the knowledge_base,
        passing the pre-validated node objects directly.
        """
        properties = relation_data.get("properties")
        caller_name = f"{self.agent.__class__.__name__}._add_new_fact"

        status = validate_and_add_relation(
            agent=self.agent,
            sub_node=sub_node,
            obj_node=obj_node,
            relation_type=relation_type,
            properties=properties,
            caller_name=caller_name,
            allow_unknowns_for_topic=allow_unknowns_for_topic,
        )
        was_learned = status in ("inserted", "replaced")
        return was_learned, status

    def learn_new_fact_autonomously(
        self, fact_sentence: str, source_topic: str | None = None
    ) -> bool:
        """
        Learns from a sentence by trying the fast SymbolicParser first, then
        falling back to the UniversalInterpreter's decomposition.
        """
        caller_name = f"{self.agent.__class__.__name__}.learn_new_fact_autonomously"
        logger.info(
            "[yellow]   - Attempting to learn fact: '%s'[/yellow]", fact_sentence
        )

        # --- FIX: Try the SymbolicParser first ---
        symbolic_interpretations = self.agent.parser.parse(fact_sentence)

        atomic_relations: list[RelationData] = []
        provenance = "symbolic_parser"

        if symbolic_interpretations:
            logger.info("   - ✅ Understood via Symbolic Parser.")
            for interp in symbolic_interpretations:
                if relation := interp.get("relation"):
                    atomic_relations.append(relation)
        else:
            # --- Fallback to the UniversalInterpreter ---
            logger.warning(
                "  [Cognitive Flow]: Symbolic parsing failed. Falling back to LLM interpreter."
            )
            atomic_relations = self.agent.interpreter.decompose_sentence_to_relations(
                text=fact_sentence, main_topic=source_topic
            )
            provenance = "llm_decomposition"

        if not atomic_relations:
            logger.warning(
                "[warning]Interpreter could not extract any atomic facts from the sentence.[/warning] (in %s)",
                caller_name,
            )
            return False

        facts_learned_count = 0
        for relation_data in atomic_relations:
            props = relation_data.setdefault("properties", {})
            props.setdefault(
                "confidence", 0.9 if provenance == "symbolic_parser" else 0.8
            )
            props.setdefault("provenance", provenance)
            logger.info("[green]   - Interpreted Relation: %s[/green]", relation_data)

            success, _ = self._process_statement_for_learning(
                relation_data, allow_unknowns_for_topic=source_topic
            )
            if success:
                facts_learned_count += 1

        if facts_learned_count > 0:
            logger.info(
                "[green]  [Autonomous Learning]: Successfully learned and saved %d new fact(s).[/green]",
                facts_learned_count,
            )
            return True

        logger.info(
            "   - No new facts were learned from the decomposed relations (e.g., trivial, invalid)."
        )
        return False

    def _add_or_update_concept_quietly(
        self,
        name: str,
        node_type: str = "concept",
    ) -> ConceptNode | None:
        """Find a concept node by name, creating it if it doesn't exist."""
        clean_name = self.agent._clean_phrase(name)
        if not clean_name:
            return None
        node = self.agent.graph.get_node_by_name(clean_name)
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
            node = self.agent.graph.add_node(
                ConceptNode(clean_name, node_type=determined_type)
            )
        return node

    def manual_add_knowledge_quietly(
        self,
        concept_name1: str,
        concept_type1: str,  # This argument is now effectively ignored for node creation
        relation: str,
        concept_name2: str,
        weight: float = 0.5,
    ) -> None:
        """Add a structured fact directly to the knowledge graph WITHOUT logging."""
        # --- FIX: Use the smart constructor with a default node type ---
        # This ignores the flawed 'concept_type1' from the seed data for node creation.
        node1 = self._add_or_update_concept_quietly(concept_name1, node_type="concept")
        node2 = self._add_or_update_concept_quietly(concept_name2, node_type="concept")

        if node1 and node2:
            self.agent.graph.add_edge(node1, node2, relation, weight)
