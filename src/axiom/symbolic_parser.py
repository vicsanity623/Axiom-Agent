from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .universal_interpreter import InterpretData, RelationData

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent


class SymbolicParser:
    """A deterministic, rule-based parser for understanding simple language.

    This class is the core of the agent's native language understanding.
    It attempts to deconstruct a sentence into a structured `InterpretData`
    object by applying a series of grammatical rules. It relies on the
    agent's `LexiconManager` to identify parts of speech.

    This parser is designed to be the first-pass interpreter, allowing the
    agent to bypass the LLM for sentences it can understand on its own.
    """

    PREPOSITION_PATTERN = re.compile(
        r"^(?P<subject>.+?)\s+"
        r"(?P<verb>is|are|was|were)\s+"
        r"(?P<object>.+?)\s+"
        r"(?P<preposition>in|on|at|of|from|with|inside)\s+"
        r"(?P<prep_object>.+?)$",
        re.IGNORECASE,
    )

    PREPOSITION_TO_RELATION_MAP = {
        ("is", "in"): "is_located_in",
        ("are", "in"): "is_located_in",
        ("is", "on"): "is_located_on",
        ("are", "on"): "is_located_on",
        ("is", "at"): "is_located_at",
        ("are", "at"): "is_located_at",
    }

    def __init__(self, agent: CognitiveAgent):
        """Initialize the SymbolicParser.

        Args:
            agent: The instance of the CognitiveAgent this parser will serve.
        """
        self.agent = agent
        print("   - Symbolic Parser initialized.")

    def parse(self, text: str) -> InterpretData | None:
        """Attempt to parse a simple sentence into a structured intent.

        This method applies a prioritized sequence of rules to deconstruct
        the input text:
        1.  Checks for questions (e.g., "what is...").
        2.  Checks for simple commands (e.g., "show all facts").
        3.  Checks for Subject-Verb-Adjective structure.
        4.  Falls back to a general Subject-Verb-Object structure.

        If a rule matches, it returns a structured `InterpretData` object.
        If no rules match, it returns None, signaling a parse failure.

        Args:
            text: The user input string to parse.

        Returns:
            An `InterpretData` object on a successful parse, or None.
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

        preposition_match = self.PREPOSITION_PATTERN.match(text)
        if preposition_match:
            groups = preposition_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb = groups["verb"].lower()
            object_phrase = self.agent._clean_phrase(groups["object"])
            preposition = groups["preposition"].lower()
            prep_object = self.agent._clean_phrase(groups["prep_object"])

            relation_type = self.PREPOSITION_TO_RELATION_MAP.get((verb, preposition))

            if relation_type:
                print(
                    f"  [Symbolic Parser]: Successfully parsed Prepositional structure: '{subject}' -> '{relation_type}' -> '{prep_object}'.",
                )
                relation = RelationData(
                    subject=subject, verb=relation_type, object=prep_object,
                )
                return InterpretData(
                    intent="statement_of_fact",
                    entities=[
                        {"name": subject, "type": "CONCEPT"},
                        {"name": prep_object, "type": "CONCEPT"},
                        {"name": object_phrase, "type": "CONCEPT"},
                    ],
                    relation=relation,
                    key_topics=[subject, prep_object, object_phrase],
                    full_text_rephrased=text,
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
        """Scan a list of words to find a single, known verb.

        This helper function iterates through the words of a sentence and
        uses the `LexiconManager` to check if any of them are categorized
        as a 'verb'.

        For the current V1 parser, it will only succeed if exactly one
        known verb is found, as this simplifies the parsing logic.

        Args:
            words: A list of tokenized words from the input sentence.

        Returns:
            A tuple containing the verb string and its index, or None if
            zero or more than one verb is found.
        """
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
        """Check if a word is categorized as a specific part of speech.

        This method queries the agent's Lexicon to determine if a word
        has an "is_a" relationship to a given part-of-speech concept
        (e.g., checks if 'brown' -> 'is_a' -> 'adjective').

        Args:
            word: The word to check.
            pos: The part of speech to check for (e.g., "verb", "adjective").

        Returns:
            True if the word is categorized as the given part of speech,
            False otherwise.
        """
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
