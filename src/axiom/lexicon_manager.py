from __future__ import annotations

# src/axiom/lexicon_manager.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent


class LexiconManager:
    """Manages the agent's knowledge about words (its internal dictionary)."""

    __slots__ = ("agent",)

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initializes the LexiconManager."""
        self.agent = agent
        print("   - Lexicon Manager initialized.")

    def is_known_word(self, word: str) -> bool:
        """Checks if a word exists as a concept in the knowledge graph."""
        clean_word = self.agent._clean_phrase(word)
        if not clean_word:
            return False
        return self.agent.graph.get_node_by_name(clean_word) is not None

    def add_linguistic_knowledge(
        self,
        word: str,
        part_of_speech: str,
        definition: str | None = None,
    ) -> None:
        """Adds a new word and its linguistic properties to the graph."""
        clean_word = self.agent._clean_phrase(word)
        if not clean_word:
            return

        word_node = self.agent._add_or_update_concept(clean_word)
        pos_node = self.agent._add_or_update_concept(part_of_speech)

        if word_node and pos_node:
            self.agent.graph.add_edge(word_node, pos_node, "is_a", weight=0.95)
            print(
                f"    [Lexicon]: Learned that '{clean_word}' is a '{part_of_speech}'.",
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
                print(f"    [Lexicon]: Added definition for '{clean_word}'.")
