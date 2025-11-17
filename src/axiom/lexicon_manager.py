from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    ) -> None:
        """
        Adds a new linguistic fact directly to the knowledge graph as a
        standard 'is_a' relationship. This is the primary method for
        updating the lexicon.
        """
        clean_word = self.agent._clean_phrase(word)
        pos_lower = part_of_speech.lower()
        if not clean_word:
            return

        word_node = self.agent.knowledge_manager.get_or_create_concept(
            clean_word, node_type=pos_lower
        )
        pos_node = self.agent.knowledge_manager.get_or_create_concept(
            pos_lower, node_type="part_of_speech"
        )

        if word_node and pos_node:
            # Add the fact directly to the graph
            validate_and_add_relation(
                agent=self.agent,  # Pass the agent instance
                sub_node=word_node,
                obj_node=pos_node,
                relation_type="is_a",
                properties=PropertyData(provenance=provenance, confidence=confidence),
            )
            logger.info(
                "    [Lexicon]: Learned that '%s' is a '%s'.",
                clean_word,
                pos_lower,
            )
