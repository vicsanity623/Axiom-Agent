from __future__ import annotations

import re
from typing import TYPE_CHECKING

from axiom.universal_interpreter import InterpretData, RelationData

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent


class SymbolicParser:
    """A deterministic, rule-based parser for understanding simple language.

    This class is the core of the agent's native language understanding.
    It attempts to deconstruct a sentence into a structured `InterpretData`
    object by applying a series of grammatical rules. It relies on the
    agent's `LexiconManager` to identify parts of speech.

    This parser is designed to be the first-pass interpreter, allowing the
    agent to bypass the LLM for sentences it can understand on its own.
    """

    __slots__ = (
        "agent",
        "RELATIONAL_QUESTION_PATTERN",
        "PREPOSITION_PATTERN",
        "QUESTION_WORDS",
        "PREPOSITION_TO_RELATION_MAP",
    )

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initialize the SymbolicParser."""
        self.RELATIONAL_QUESTION_PATTERN = re.compile(
            r"(?i)^(is|are|was|were|do|does|did|has|have|had)\s+"
            r"(?P<subject>.+?)\s+"
            r"(?P<verb>\w+)\s+"
            r"(?P<object>.+)\?*$",
        )
        self.PREPOSITION_PATTERN = re.compile(
            r"(?i)^(?P<subject>.+?)\s+"
            r"(?P<verb>is|are|was|were)\s+"
            r"(?P<object>.+?)\s+"
            r"(?P<preposition>in|on|at|of|from|with|inside)\s+"
            r"(?P<prep_object>.+?)$",
        )
        self.QUESTION_WORDS = {
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "which",
            "whomst",
        }
        self.PREPOSITION_TO_RELATION_MAP = {
            ("is", "in"): "is_located_in",
            ("are", "in"): "is_located_in",
            ("is", "on"): "is_located_on",
            ("are", "on"): "is_located_on",
            ("is", "at"): "is_located_at",
            ("are", "at"): "is_located_at",
        }
        self.agent = agent
        print("   - Symbolic Parser initialized.")

    def _split_into_clauses(self, text: str) -> list[str]:
        """Splits a complex text block into simpler, independent clauses."""

        sentences = [s.strip() for s in text.split(".") if s.strip()]

        all_clauses = []
        split_phrases = [" and ", " but ", " which ", " who ", " that "]
        placeholder = "||CLAUSE_BREAK||"

        for sentence in sentences:
            clause_text = sentence
            for phrase in split_phrases:
                clause_text = re.sub(
                    re.escape(phrase),
                    placeholder,
                    clause_text,
                    flags=re.IGNORECASE,
                )

            clauses_from_sentence = [
                c.strip(" ,") for c in clause_text.split(placeholder) if c.strip()
            ]
            all_clauses.extend(clauses_from_sentence)

        if len(all_clauses) > 1:
            print(
                f"  [Chunker]: Split text into {len(all_clauses)} clauses: {all_clauses}",
            )

        return all_clauses

    def parse(self, text: str) -> list[InterpretData] | None:
        """
        Attempts to parse a sentence by running it through a multi-stage pipeline.
        Returns a list of interpretations (one for each successfully parsed clause).
        """
        print(f"  [Symbolic Parser]: Attempting to parse sentence: '{text}'")

        clauses = self._split_into_clauses(text)

        all_interpretations: list[InterpretData] = []

        for clause in clauses:
            interpretation = self._parse_single_clause(clause)
            if interpretation:
                all_interpretations.append(interpretation)

        if all_interpretations:
            return all_interpretations

        print("  [Symbolic Parser]: Failed. No clauses could be parsed.")
        return None

    # --- YOUR EXISTING LOGIC, NOW IN A HELPER METHOD ---
    def _parse_single_clause(self, clause: str) -> InterpretData | None:
        """Applies a prioritized sequence of rules to deconstruct a single clause."""
        words = clause.lower().split()
        if not words:
            return None

        # Rule 1: Relational Questions
        relational_match = self.RELATIONAL_QUESTION_PATTERN.match(clause)
        if relational_match:
            groups = relational_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb = groups["verb"].lower()
            object_ = self.agent._clean_phrase(groups["object"])

            print(
                f"  [Symbolic Parser]: Successfully parsed Relational Question: '{subject}' --[{verb}]--> '{object_}'?",
            )
            relation = RelationData(subject=subject, verb=verb, object=object_)
            return InterpretData(
                intent="question_about_concept",
                entities=[
                    {"name": subject, "type": "CONCEPT"},
                    {"name": object_, "type": "CONCEPT"},
                ],
                relation=relation,
                key_topics=[subject, object_],
                full_text_rephrased=clause,
            )

        # Rule 2: Wh-Questions
        if words[0] in self.QUESTION_WORDS:
            print("  [Symbolic Parser]: Successfully parsed a wh-question.")
            entity_name = " ".join(words[2:]) if len(words) > 2 else " ".join(words[1:])
            entity_name = entity_name.replace("?", "").strip()
            entity_name = re.sub(r"^(is|are|was|were)\s+", "", entity_name)
            return InterpretData(
                intent="question_about_entity",
                entities=[{"name": entity_name, "type": "CONCEPT"}],
                relation=None,
                key_topics=[entity_name],
                full_text_rephrased=clause,
            )

        # Rule 3: Commands
        if clause.lower() == "show all facts":
            print("  [Symbolic Parser]: Successfully parsed 'show all facts' command.")
            return InterpretData(
                intent="command",
                entities=[],
                relation=None,
                key_topics=["show all facts"],
                full_text_rephrased="User issued a command to show all facts.",
            )

        # Rule 4: Prepositional Phrases
        preposition_match = self.PREPOSITION_PATTERN.match(clause)
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
                    subject=subject,
                    verb=relation_type,
                    object=prep_object,
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
                    full_text_rephrased=clause,
                )

        # Rule 5: Statement Structures
        verb_info = self._find_verb(words)
        if not verb_info:
            return None
        verb, verb_index = verb_info

        # Rule 5a: Adjectives
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
                    full_text_rephrased=clause,
                )

        # Rule 5b: Subject-Verb-Object
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
                full_text_rephrased=clause,
            )

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
