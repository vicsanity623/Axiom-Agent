from __future__ import annotations

from typing import TYPE_CHECKING

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
        if not words:
            return None

        question_words = {"what", "who", "where", "when", "why", "how", "is", "are"}
        if words[0] in question_words:
            print("  [Symbolic Parser]: Successfully parsed a question.")
            entity_name = " ".join(words[2:]) if len(words) > 2 else " ".join(words[1:])
            entity_name = entity_name.replace("?", "").strip()
            return InterpretData(
                intent="question_about_entity",
                entities=[{"name": entity_name, "type": "CONCEPT"}],
                relation=None,
                key_topics=[entity_name],
                full_text_rephrased=text,
            )

        if text.lower() == "show all facts":
            print("  [Symbolic Parser]: Successfully parsed 'show all facts' command.")
            return InterpretData(
                intent="command",
                entities=[],
                relation=None,
                key_topics=["show all facts"],
                full_text_rephrased="User issued a command to show all facts.",
            )

        verb_info = self._find_verb(words)
        if not verb_info:
            print(
                "  [Symbolic Parser]: Failed. Could not identify a single known verb.",
            )
            return None

        verb, verb_index = verb_info

        if len(words) > verb_index + 1:
            potential_adjective = words[verb_index + 1]
            if self._is_part_of_speech(potential_adjective, "adjective"):
                subject = " ".join(words[:verb_index])
                subject = self.agent._clean_phrase(subject)

                print(
                    f"  [Symbolic Parser]: Successfully parsed S-V-Adjective structure: '{subject}' has property '{potential_adjective}'.",
                )

                relation = RelationData(
                    subject=subject,
                    verb="has_property",
                    object=potential_adjective,
                )
                return InterpretData(
                    intent="statement_of_fact",
                    entities=[
                        {"name": subject, "type": "CONCEPT"},
                        {"name": potential_adjective, "type": "PROPERTY"},
                    ],
                    relation=relation,
                    key_topics=[subject, potential_adjective],
                    full_text_rephrased=text,
                )

        if verb_index > 0 and verb_index < len(words) - 1:
            subject = " ".join(words[:verb_index])
            object_ = " ".join(words[verb_index + 1 :])
            subject = self.agent._clean_phrase(subject)
            object_ = self.agent._clean_phrase(object_)

            print(
                f"  [Symbolic Parser]: Successfully parsed S-V-O structure: '{subject}' -> '{verb}' -> '{object_}'.",
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

        print("  [Symbolic Parser]: Failed. Sentence structure not recognized.")
        return None

    def _find_verb(self, words: list[str]) -> tuple[str, int] | None:
        """Scans a list of words to find a single, known verb."""
        found_verbs = []
        for i, word in enumerate(words):
            word_node = self.agent.graph.get_node_by_name(word)
            if word_node:
                if self._is_part_of_speech(word, "verb"):
                    found_verbs.append((word, i))

        if len(found_verbs) == 1:
            return found_verbs[0]

        return None

    def _is_part_of_speech(self, word: str, pos: str) -> bool:
        """Checks if a word has a specific part of speech in the Lexicon."""
        word_node = self.agent.graph.get_node_by_name(word)
        if not word_node:
            return False

        is_a_edges = [
            edge
            for edge in self.agent.graph.get_edges_from_node(word_node.id)
            if edge.type == "is_a"
        ]

        for edge in is_a_edges:
            target_node = self.agent.graph.get_node_by_id(edge.target)
            if target_node and target_node.name == pos:
                return True
        return False
