from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .universal_interpreter import PropertyData

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class LexiconManager:
    """Manage the agent's knowledge about words (its internal dictionary).

    This class provides a dedicated API for interacting with the agent's
    linguistic knowledge, which is stored in the main knowledge graph. It
    handles the creation of word concepts, their connections to parts of
    speech, definitions, and supports polysemy (multiple meanings per word).
    """

    __slots__ = ("agent",)

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initialize the LexiconManager.

        Args:
            agent: The instance of the CognitiveAgent this manager will serve.
        """
        self.agent = agent
        logger.info("[success]Lexicon Manager initialized.[success]")

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
            logger.debug(
                "Failed to record lexical observation for %s:%s",
                word,
                pos,
                exc_info=True,
            )

    def is_promoted_word(self, word: str) -> bool:
        """Return True only if the word exists and has at least one trusted part of speech."""
        clean = self.agent._clean_phrase(word)
        if not clean:
            return False
        node = self.agent.graph.get_node_by_name(clean)
        if not node:
            return False
        props = self.agent.graph.graph.nodes[node.id].get("properties", {})
        promoted_as = props.get("lexical_promoted_as")
        return isinstance(promoted_as, dict) and bool(promoted_as)

    def get_promoted_pos(self, word: str) -> dict[str, float]:
        """
        Returns a dictionary of promoted parts of speech for a word and their confidences.
        Supports polysemy.
        """
        clean = self.agent._clean_phrase(word)
        if not clean:
            return {}
        node = self.agent.graph.get_node_by_name(clean)
        if not node:
            return {}
        props = self.agent.graph.graph.nodes[node.id].get("properties", {})
        promoted_as = props.get("lexical_promoted_as", {})
        if isinstance(promoted_as, dict):
            return promoted_as
        return {}

    def prune_low_confidence_observations(self, threshold: float = 0.3) -> int:
        """
        Iterate through all concepts and remove lexical observations below a confidence threshold.
        This is intended to be called during a refinement cycle.

        Args:
            threshold: The confidence score below which observations will be removed.

        Returns:
            The number of observations pruned.
        """
        pruned_count = 0
        nodes_to_check = list(self.agent.graph.graph.nodes(data=True))
        for node_id, data in nodes_to_check:
            props = data.get("properties", {})
            observations = props.get("lexical_observations")
            if isinstance(observations, dict):
                for pos, confidences in list(observations.items()):
                    surviving_confidences = [
                        conf for conf in confidences if conf >= threshold
                    ]
                    if surviving_confidences:
                        if len(surviving_confidences) < len(confidences):
                            pruned_count += len(confidences) - len(
                                surviving_confidences
                            )
                            observations[pos] = surviving_confidences
                    else:
                        pruned_count += len(confidences)
                        del observations[pos]
        if pruned_count > 0:
            logger.info(
                "[Refinement]: Pruned %d low-confidence lexical observations.",
                pruned_count,
            )
        return pruned_count

    def _promote_word_for_test(self, word: str, pos: str) -> None:
        """
        A helper method ONLY for tests. It directly promotes a word to the
        lexicon, bypassing the normal observation and confidence checks.
        Supports polysemy by adding/updating the POS in the promoted dictionary.
        """
        node = self.agent._add_or_update_concept_quietly(word)
        if not node:
            return

        props = self.agent.graph.graph.nodes[node.id].setdefault("properties", {})
        promoted_as = props.setdefault("lexical_promoted_as", {})
        if isinstance(promoted_as, dict):
            promoted_as[pos] = 1.0
        props["lexical_promoted_confidence"] = 1.0

    def promote_word(self, word: str, pos: str, confidence: float = 0.95) -> None:
        """Wrapper for knowledge_base.promote_word() for consistency."""
        try:
            from . import knowledge_base as kb

            kb.promote_word(self.agent, word, pos, confidence)
        except Exception:
            logger.exception("Failed to promote word: %s", word)
