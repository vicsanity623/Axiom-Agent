from __future__ import annotations

import logging
import typing
from typing import TYPE_CHECKING, Any

from .graph_core import ConceptNode
from .knowledge_base import validate_and_add_relation
from .universal_interpreter import PropertyData

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class LexiconManager:
    """
    Manages the agent's knowledge about words using the main knowledge graph.
    This "Knowledge-First" approach treats grammatical facts (e.g., 'cat is a noun')
    as first-class knowledge, making them instantly available and reason-able.
    """

    __slots__ = ("agent",)

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initialize the LexiconManager."""
        self.agent = agent
        logger.info("[green]Lexicon Manager initialized.[/green]")

    def is_known_word(self, word: str) -> bool:
        """Check if a word exists as a concept in the knowledge graph."""
        clean_word = self.agent._clean_phrase(word)
        if not clean_word:
            return False
        return self.agent.graph.get_node_by_name(clean_word) is not None

    def is_part_of_speech(self, word: str, pos: str) -> bool:
        """
        Checks if the word has a direct 'is_a' relationship to the given
        part of speech in the knowledge graph.
        """
        clean_word = self.agent._clean_phrase(word)
        pos_lower = pos.lower()
        if not clean_word:
            return False

        word_node = self.agent.graph.get_node_by_name(clean_word)
        if not word_node:
            return False

        # Query the graph directly for the relationship
        for edge in self.agent.graph.get_edges_from_node(word_node.id):
            if edge.type == "is_a":
                target_node = self.agent.graph.get_node_by_id(edge.target)
                if target_node and target_node.name == pos_lower:
                    return True
        return False

    def add_linguistic_knowledge(
        self,
        word: str,
        part_of_speech: str,
        confidence: float = 0.9,
        provenance: str = "seed",
    ) -> ConceptNode | None:
        """
        The single, authoritative method for creating or updating a concept in the
        knowledge graph. It ensures that every concept has lexical properties.
        """
        clean_word = self.agent._clean_phrase(word)
        pos_lower = self.agent._clean_phrase(part_of_speech)
        if not clean_word or not pos_lower:
            return None

        # Step 1: Get or Create the main word node
        node = self.agent.graph.get_node_by_name(clean_word)
        if node is None:
            node = self.agent.graph.add_node(
                ConceptNode(name=clean_word, node_type=pos_lower)
            )
            logger.info(
                "   - Added new concept to graph: %s (%s)", clean_word, pos_lower
            )

        # Ensure the main node has its lexical property
        if node.properties is None:
            node.properties = {}  # type: ignore

        # Cast to dict to allow arbitrary keys (bypassing PropertyData TypedDict limits)
        props_dict = typing.cast("dict[str, Any]", node.properties)
        promoted_as = props_dict.setdefault("lexical_promoted_as", {})
        if isinstance(promoted_as, dict):
            promoted_as[pos_lower] = max(promoted_as.get(pos_lower, 0.0), confidence)

        # Update the node in the graph
        self.agent.graph.add_node(node)

        # Step 2: Get or Create the part-of-speech node itself
        pos_node: ConceptNode | None
        if clean_word == pos_lower:
            pos_node = node
        else:
            pos_node = self.agent.graph.get_node_by_name(pos_lower)
            if pos_node is None:
                # Recursively call this function to create the POS node
                self.add_linguistic_knowledge(
                    pos_lower, "part_of_speech", confidence=1.0, provenance="system"
                )
                pos_node = self.agent.graph.get_node_by_name(pos_lower)

        if pos_node is None:
            logger.error("Failed to create part-of-speech node '%s'.", pos_lower)
            return node

        # Step 3: Add the 'is_a' relationship
        validate_and_add_relation(
            agent=self.agent,
            sub_node=node,
            obj_node=pos_node,
            relation_type="is_a",
            properties=PropertyData(provenance=provenance, confidence=confidence),
        )
        logger.info(
            "    [Lexicon]: Learned that '%s' is a '%s'.", clean_word, pos_lower
        )

        return node
