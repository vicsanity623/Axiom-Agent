# src/axiom/symbolic_parser.py

from __future__ import annotations

from typing import TYPE_CHECKING

# We will reuse the same data structures as the LLM interpreter
# to make the integration seamless.
from .universal_interpreter import InterpretData, RelationData

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent


class SymbolicParser:
    def __init__(self, agent: CognitiveAgent):
        self.agent = agent
        print("   - Symbolic Parser initialized.")

    def parse(self, text: str) -> InterpretData | None:
        """
        Attempts to parse a simple sentence into a structured intent.
        Returns an InterpretData object on success, or None on failure.
        """
        print("  [Symbolic Parser]: Attempting to parse sentence...")

        words = text.lower().split()

        # --- Rule 1: Handle simple commands ---
        if text.lower() == "show all facts":
            print("  [Symbolic Parser]: Successfully parsed 'show all facts' command.")
            return InterpretData(
                intent="command",
                entities=[],
                relation=None,
                key_topics=["show all facts"],
                full_text_rephrased="User issued a command to show all facts.",
            )

        # --- Rule 2: Find a single, known verb ---
        verb_info = self._find_verb(words)
        if not verb_info:
            print(
                "  [Symbolic Parser]: Failed. Could not identify a single known verb."
            )
            return None

        verb, verb_index = verb_info

        # --- Rule 3: Identify Subject and Object ---
        if verb_index == 0 or verb_index == len(words) - 1:
            print("  [Symbolic Parser]: Failed. Verb is at start/end of sentence.")
            return None

        subject = " ".join(words[:verb_index])
        object_ = " ".join(words[verb_index + 1 :])

        # Clean up articles for cleaner concepts
        subject = self.agent._clean_phrase(subject)
        object_ = self.agent._clean_phrase(object_)

        print(
            f"  [Symbolic Parser]: Successfully parsed S-V-O structure: '{subject}' -> '{verb}' -> '{object_}'."
        )

        relation = RelationData(subject=subject, verb=verb, object=object_)

        return InterpretData(
            intent="statement_of_fact",
            entities=[
                {"name": subject, "type": "CONCEPT"},
                {"name": object_, "type": "CONCEPT"},
            ],
            relation=relation,
            key_topics=[subject, object_],
            full_text_rephrased=text,
        )

    def _find_verb(self, words: list[str]) -> tuple[str, int] | None:
        """Scans a list of words to find a single, known verb."""
        found_verbs = []
        for i, word in enumerate(words):
            word_node = self.agent.graph.get_node_by_name(word)
            if word_node:
                is_a_edges = [
                    edge
                    for edge in self.agent.graph.get_edges_from_node(word_node.id)
                    if edge.type == "is_a"
                ]

                for edge in is_a_edges:
                    target_node = self.agent.graph.get_node_by_id(edge.target)

                    if target_node and target_node.name == "verb":
                        found_verbs.append((word, i))
                        break

        if len(found_verbs) == 1:
            return found_verbs[0]

        return None
