from __future__ import annotations

# src/axiom/lexicon_manager.py
import logging
from typing import TYPE_CHECKING

from .universal_interpreter import PropertyData

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent


class LexiconManager:
    """Manage the agent's knowledge about words (its internal dictionary).

    This class provides a dedicated API for interacting with the agent's
    linguistic knowledge, which is stored in the main knowledge graph. It
    handles the creation of word concepts and their connections to parts of
    speech and definitions.
    """

    __slots__ = ("agent",)

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initialize the LexiconManager.

        Args:
            agent: The instance of the CognitiveAgent this manager will serve.
        """
        self.agent = agent
        print("   - Lexicon Manager initialized.")

    def is_known_word(self, word: str) -> bool:
        """Check if a word exists as a concept in the knowledge graph.

        Args:
            word: The word to check.

        Returns:
            True if the word is known, False otherwise.
        """

        if word.lower() in ["a", "an", "the", "i"]:
            return self.agent.graph.get_node_by_name(word.lower()) is not None

        clean_word = self.agent._clean_phrase(word)
        if not clean_word:
            return False
        return self.agent.graph.get_node_by_name(clean_word) is not None

    def add_linguistic_knowledge_quietly(
        self,
        word: str,
        part_of_speech: str,
        definition: str | None = None,
    ) -> None:
        """Add a new word and its linguistic properties WITHOUT printing.

        Used for high-volume initial vocabulary seeding.

        This method creates the necessary nodes and edges to represent a
        word's meaning. It links the word to its part of speech (e.g.,
        'dog' -> 'is_a' -> 'noun') and optionally to its definition.

        It also contains special logic to create higher-level semantic
        links, such as categorizing 'adjective' as a type of 'property'.

        Args:
            word: The word to learn.
            part_of_speech: The grammatical role of the word (e.g., 'noun').
            definition: An optional definition string for the word.
        """

        if word.lower() in ["a", "an", "the", "i"]:
            clean_word = word.lower()
        else:
            clean_word = self.agent._clean_phrase(word)

        if not clean_word:
            return

        word_node = self.agent._add_or_update_concept_quietly(clean_word)
        pos_node = self.agent._add_or_update_concept_quietly(part_of_speech)

        if word_node and pos_node:
            self.agent.graph.add_edge(
                word_node,
                pos_node,
                "is_a",
                weight=0.95,
                properties=PropertyData(provenance="seed", confidence=0.95),
            )

            from axiom import knowledge_base as kb

            kb.promote_word(self.agent, clean_word, part_of_speech, confidence=0.95)

            if part_of_speech == "adjective":
                property_node = self.agent._add_or_update_concept("property")
                if property_node:
                    self.agent.graph.add_edge(
                        pos_node,
                        property_node,
                        "is_a",
                        weight=0.9,
                    )

        if definition:
            def_node = self.agent._add_or_update_concept(definition)
            if word_node and def_node:
                self.agent.graph.add_edge(
                    word_node,
                    def_node,
                    "has_definition",
                    weight=0.9,
                )

    def observe_word_pos(self, word: str, pos: str, confidence: float = 0.5) -> None:
        """
        Record a POS observation for `word`. Delegates to knowledge_base.record_lexical_observation.
        """
        try:
            from . import knowledge_base as kb

            kb.record_lexical_observation(self.agent, word, pos, confidence)
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to record lexical observation for %s:%s",
                word,
                pos,
                exc_info=True,
            )

    def is_promoted_word(self, word: str) -> bool:
        """Return True only if the word exists and was promoted to the lexicon (trusted)."""
        clean = self.agent._clean_phrase(word)
        if not clean:
            return False
        node = self.agent.graph.get_node_by_name(clean)
        if not node:
            return False
        props = self.agent.graph.graph.nodes[node.id].get("properties", {})
        return bool(props.get("lexical_promoted_as"))

    def _promote_word_for_test(self, word: str, pos: str) -> None:
        """
        A helper method ONLY for tests. It directly promotes a word to the
        lexicon, bypassing the normal observation and confidence checks.
        """
        node = self.agent._add_or_update_concept_quietly(word)
        if not node:
            return

        props = self.agent.graph.graph.nodes[node.id].setdefault("properties", {})
        props["lexical_promoted_as"] = pos
        props["lexical_promoted_confidence"] = 1.0
