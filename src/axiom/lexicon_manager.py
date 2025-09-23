from __future__ import annotations

# src/axiom/lexicon_manager.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent


class LexiconManager:
    """Manage the agent's knowledge about words (its internal dictionary).

    This class provides a dedicated API for interacting with the agent's
    linguistic knowledge, which is stored in the main knowledge graph. It
    handles the creation of word concepts and their connections to parts of
    speech and definitions.
    """

    __slots__ = ("agent",)

    def __init__(self, agent: "CognitiveAgent") -> None:
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
        """Add a new word and its linguistic properties to the graph.

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

            if part_of_speech == "adjective":
                property_node = self.agent._add_or_update_concept("property")
                if property_node:
                    self.agent.graph.add_edge(
                        pos_node, property_node, "is_a", weight=0.9
                    )
                    print(
                        "    [Lexicon]: Categorized 'adjective' as a type of 'property'.",
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
