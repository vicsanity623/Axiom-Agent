from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, cast

from nltk.stem import WordNetLemmatizer

from .constants import PROVENANCE_RANK
from .dictionary_utils import get_word_info_from_wordnet
from .graph_core import ConceptNode
from .knowledge_base import validate_and_add_relation

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent
    from .universal_interpreter import RelationData

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()


class KnowledgeManager:
    def __init__(self, agent: CognitiveAgent):
        """Initializes the KnowledgeManager with a reference to the main agent."""
        self.agent = agent

    def _process_statement_for_learning(
        self,
        relation: RelationData,
        allow_unknowns_for_topic: str | None = None,
    ) -> tuple[bool, str]:
        """
        Process a structured fact, with robust, centralized logic for handling
        homonyms before validation and belief revision.
        """
        if self.agent.inference_mode:
            return False, "Agent is in read-only mode."

        subject_name_or_none = self.agent._extract_name_from_relation(
            relation.get("subject")
        )
        object_name_or_none = self.agent._extract_name_from_relation(
            relation.get("object")
        )
        verb_name_or_none = self.agent._extract_name_from_relation(relation.get("verb"))

        if not all((subject_name_or_none, object_name_or_none, verb_name_or_none)):
            return False, "Incomplete fact structure."

        subject_name = cast("str", subject_name_or_none)
        object_name = cast("str", object_name_or_none)
        verb_name = cast("str", verb_name_or_none)

        if subject_name == object_name:
            return False, "trivial_fact"

        logger.info(
            "[cyan]   - Processing: %s -> %s -> %s[/cyan]",
            subject_name,
            verb_name,
            object_name,
        )

        relation_type = self.agent.get_relation_type(
            verb_name, subject_name, object_name
        )

        obj_node = self._add_or_update_concept(object_name)
        if not obj_node:
            return False, f"Failed to create concept for object: '{object_name}'."

        sub_node = self._add_or_update_concept(
            subject_name,
            context_relation_type=relation_type,
            context_object_name=object_name,
        )
        if not sub_node:
            return False, f"Failed to create concept for subject: '{subject_name}'."

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
        A simple wrapper that delegates fact addition to the knowledge_base.
        """
        candidate_fact = cast(
            "RelationData",
            {"subject": sub_node.name, "verb": relation_type, "object": obj_node.name},
        )
        properties = relation_data.get("properties")
        caller_name = f"{self.agent.__class__.__name__}._add_new_fact"
        original_subject_name = self.agent._extract_name_from_relation(
            relation_data.get("subject")
        )
        status = validate_and_add_relation(
            self.agent,
            candidate_fact,
            properties,
            caller_name=caller_name,
            allow_unknowns_for_topic=allow_unknowns_for_topic or original_subject_name,
        )
        was_learned = status in ("inserted", "replaced")
        return was_learned, status

    def learn_new_fact_autonomously(
        self, fact_sentence: str, source_topic: str | None = None
    ) -> bool:
        """Learns from a sentence by decomposing it into atomic facts and processing them."""
        caller_name = f"{self.agent.__class__.__name__}.learn_new_fact_autonomously"
        logger.info(
            "[yellow]   - Attempting to learn fact: '%s'[/yellow]", fact_sentence
        )
        atomic_relations = self.agent.interpreter.decompose_sentence_to_relations(
            text=fact_sentence, main_topic=source_topic
        )
        if not atomic_relations:
            logger.warning(
                "[warning]Interpreter could not extract any atomic facts from the sentence.[/warning] (in %s)",
                caller_name,
            )
            return False
        facts_learned_count = 0
        for relation_data in atomic_relations:
            props = relation_data.setdefault("properties", {})
            props.setdefault("confidence", 0.8)
            props.setdefault("provenance", "llm_decomposition")
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
        logger.warning(
            "[yellow]Failed to learn any new facts from the decomposed relations (e.g., trivial, invalid, not worthy).[/yellow] (in %s)",
            caller_name,
        )
        return False

    def _add_or_update_concept(
        self,
        name: str,
        node_type: str = "concept",
        context_relation_type: str | None = None,
        context_object_name: str | None = None,
    ) -> ConceptNode | None:
        """
        Finds a concept by name or creates it, integrating WordNet for type detection.
        """
        clean_name = self.agent._clean_phrase(name)
        if not clean_name:
            return None

        node = self.agent.graph.get_node_by_name(clean_name)

        if node is None:
            if " " in clean_name:
                determined_type = "noun_phrase"
            else:
                word_info_list = get_word_info_from_wordnet(clean_name)
                determined_type = (
                    word_info_list[0].get("type", node_type)
                    if word_info_list and isinstance(word_info_list[0], dict)
                    else node_type
                )
            new_node_obj = ConceptNode(clean_name, node_type=determined_type)
            node = self.agent.graph.add_node(new_node_obj)
            logger.info(
                f"[cyan]   - Added new concept to graph: {clean_name} ({node.type})[/cyan]"
            )

        if context_relation_type == "is a" and context_object_name in (
            "noun",
            "verb",
            "adjective",
        ):
            self.agent.lexicon.promote_word(
                clean_name, context_object_name, confidence=0.8
            )
            return self.agent.graph.get_node_by_id(node.id)

        return node

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
        concept_type1: str,
        relation: str,
        concept_name2: str,
        weight: float = 0.5,
    ) -> None:
        """Add a structured fact directly to the knowledge graph WITHOUT logging."""
        node1 = self._add_or_update_concept_quietly(
            concept_name1, node_type=concept_type1
        )
        node2 = self._add_or_update_concept_quietly(concept_name2)
        if node1 and node2:
            self.agent.graph.add_edge(node1, node2, relation, weight)
