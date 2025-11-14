from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .universal_interpreter import PropertyData

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class LexiconManager:
    """Manage the agent's knowledge about words (its internal dictionary)."""

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
        Check if a word has been promoted as a specific part of speech.
        """
        promoted_parts = self.get_promoted_pos(word)
        return pos.lower() in promoted_parts

    def add_linguistic_knowledge_quietly(
        self,
        word: str,
        part_of_speech: str,
        definition: str | None = None,
    ) -> None:
        """Add a new word and its linguistic properties during seeding."""
        clean_word = self.agent._clean_phrase(word)
        if not clean_word:
            return

        word_node = self.agent.knowledge_manager._add_or_update_concept_quietly(
            clean_word
        )
        pos_node = self.agent.knowledge_manager._add_or_update_concept_quietly(
            part_of_speech
        )

        if word_node and pos_node:
            self.agent.graph.add_edge(
                word_node,
                pos_node,
                "is_a",
                weight=0.95,
                properties=PropertyData(provenance="seed", confidence=0.95),
            )
            self.promote_word(clean_word, part_of_speech, confidence=0.95, quiet=True)

            if part_of_speech == "adjective":
                property_node = (
                    self.agent.knowledge_manager._add_or_update_concept_quietly(
                        "property"
                    )
                )
                if property_node:
                    self.agent.graph.add_edge(
                        pos_node, property_node, "is_a", weight=0.9
                    )

        if definition:
            def_node = self.agent.knowledge_manager._add_or_update_concept_quietly(
                definition
            )
            if word_node and def_node:
                self.agent.graph.add_edge(
                    word_node, def_node, "has_definition", weight=0.9
                )

    def observe_word_pos(self, word: str, pos: str, confidence: float = 0.5) -> None:
        """Record a POS observation for a word."""
        clean_word = self.agent._clean_phrase(word)
        if not clean_word:
            return
        node = self.agent.knowledge_manager._add_or_update_concept_quietly(clean_word)
        if not node:
            return

        props = self.agent.graph.graph.nodes[node.id].setdefault("properties", {})
        observations = props.setdefault("lexical_observations", {})
        if isinstance(observations, dict):
            pos_obs = observations.setdefault(pos, [])
            pos_obs.append(confidence)

    def is_promoted_word(self, word: str) -> bool:
        """Return True if the word has at least one trusted part of speech."""
        promoted_pos = self.get_promoted_pos(word)
        return bool(promoted_pos)

    def get_promoted_pos(self, word: str) -> dict[str, float]:
        """Returns a dictionary of promoted parts of speech for a word."""
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

    def promote_word(
        self,
        word: str,
        pos: str,
        confidence: float = 0.95,
        quiet: bool = False,
    ) -> None:
        """
        Directly promotes a word to a given part of speech by updating its
        node properties in the graph. This is the final step in lexical learning.
        """
        clean_word = self.agent._clean_phrase(word)
        if not clean_word:
            return

        node = self.agent.graph.get_node_by_name(clean_word)
        if not node:
            node = self.agent.knowledge_manager._add_or_update_concept(
                clean_word, node_type=pos
            )
            if not node:
                if not quiet:
                    logger.warning(
                        "[Lexicon]: Cannot create or find node for '%s' to promote.",
                        clean_word,
                    )
                return

        props = self.agent.graph.graph.nodes[node.id].setdefault("properties", {})

        promoted_as = props.setdefault("lexical_promoted_as", {})
        if isinstance(promoted_as, dict):
            current_confidence = promoted_as.get(pos, 0.0)
            promoted_as[pos] = max(current_confidence, confidence)
        else:
            if not quiet:
                logger.error(
                    "[Lexicon]: 'lexical_promoted_as' property for '%s' is not a dict. Cannot promote.",
                    clean_word,
                )

        props.pop("lexical_observations", None)

        if not quiet:
            logger.info(
                "    [Lexicon]: Promoted '%s' as a '%s' (confidence: %.2f).",
                clean_word,
                pos,
                promoted_as.get(pos, confidence)
                if isinstance(promoted_as, dict)
                else confidence,
            )

    def prune_low_confidence_observations(self, threshold: float = 0.3) -> int:
        """
        Iterate through all concepts and remove lexical observations below a threshold.
        """
        pruned_count = 0
        for node_id in list(self.agent.graph.graph.nodes()):
            data = self.agent.graph.graph.nodes[node_id]
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
        """A helper method ONLY for tests to directly promote a word."""
        node = self.agent.knowledge_manager._add_or_update_concept_quietly(word)
        if not node:
            return

        props = self.agent.graph.graph.nodes[node.id].setdefault("properties", {})
        promoted_as = props.setdefault("lexical_promoted_as", {})
        if isinstance(promoted_as, dict):
            promoted_as[pos] = 1.0

    def review_and_promote_observed_words(
        self, promotion_threshold: float = 0.7
    ) -> int:
        """
        Iterates through all nodes with lexical observations, calculates the most
        likely part of speech, and promotes the word if confidence exceeds a threshold.
        This is a key part of the Refinement Cycle.
        """
        promoted_count = 0
        nodes_to_review = [
            node_id
            for node_id, data in self.agent.graph.graph.nodes(data=True)
            if "lexical_observations" in data.get("properties", {})
        ]

        if not nodes_to_review:
            return 0

        logger.info(
            "[Refinement]: Reviewing %d words for lexical promotion...",
            len(nodes_to_review),
        )

        for node_id in nodes_to_review:
            node = self.agent.graph.get_node_by_id(node_id)
            if not node:
                continue

            props = self.agent.graph.graph.nodes[node.id].get("properties", {})
            observations = props.get("lexical_observations")
            if not isinstance(observations, dict):
                continue

            total_votes = 0
            pos_scores: dict[str, float] = {}
            for pos, confidences in observations.items():
                if isinstance(confidences, list):
                    score = sum(confidences)
                    pos_scores[pos] = score
                    total_votes += score

            if not pos_scores or total_votes == 0:
                continue

            winner_pos = max(pos_scores, key=lambda pos: pos_scores[pos])
            winner_score = pos_scores[winner_pos]
            confidence = winner_score / total_votes

            if confidence >= promotion_threshold:
                logger.info(
                    "   - Promoting '%s' to '%s' with confidence %.2f.",
                    node.name,
                    winner_pos,
                    confidence,
                )
                self.promote_word(
                    node.name, winner_pos, confidence=confidence, quiet=True
                )
                promoted_count += 1

        if promoted_count > 0:
            logger.info(
                "[Refinement]: Promoted %d new words to the lexicon.", promoted_count
            )

        return promoted_count
