from __future__ import annotations

# symbolic_parser.py
import logging
import re
from typing import TYPE_CHECKING

from axiom.universal_interpreter import InterpretData, RelationData

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


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
        "QUESTION_WORDS",
        "YES_NO_ADJECTIVE_PATTERN",
        "PREPOSITION_PATTERN",
        "PREPOSITION_TO_RELATION_MAP",
        "SVO_PATTERN",
        "CAPITAL_PATTERN",
    )

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initialize the SymbolicParser."""
        self.CAPITAL_PATTERN = re.compile(
            r"(?i)^(?P<city>.+?)\s+is\s+(?:the\s+)?capital\s+of\s+(?P<country>.+?)\.*$",
        )
        self.SVO_PATTERN = re.compile(
            r"(?i)^(?P<subject>.+?)\s+"
            r"(?P<verb>is|are|was|were|has|have|had)\s+"
            r"(?P<object>.+?)\.*$",
        )
        self.RELATIONAL_QUESTION_PATTERN = re.compile(
            r"(?i)^(is|are|was|were|do|does|did|has|have|had)\s+"
            r"(?P<subject>.+?)\s+"
            r"(?P<verb>\w+)\s+"
            r"(?P<object>.+)\?*$",
        )
        self.YES_NO_ADJECTIVE_PATTERN = re.compile(
            r"(?i)^(is|are|was|were)\s+"
            r"(?P<subject>.+?)\s+"
            r"(?P<adjective>\w+)\?*$",
        )
        self.PREPOSITION_PATTERN = re.compile(
            r"(?i)^(?P<subject>.+?)\s+"
            r"(?P<verb>is|are|was|were)\s+"
            r"(?:.+?\s+)?"
            r"(?P<preposition>in|on|at|part\s+of|from|with|inside)\s+"
            r"(?P<prep_object>.+?)\.*$",
            re.IGNORECASE,
        )
        self.PREPOSITION_TO_RELATION_MAP = {
            ("is", "in"): "is_located_in",
            ("are", "in"): "is_located_in",
            ("is", "on"): "is_on_top_of",
            ("are", "on"): "is_on_top_of",
            ("is", "at"): "is_located_at",
            ("are", "at"): "is_located_at",
            ("is", "part of"): "is_part_of",
            ("are", "part of"): "is_part_of",
        }
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

        self.agent = agent
        logger.info("   - Symbolic Parser initialized.")

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
            logger.debug(
                "  [Chunker]: Split text into %s clauses: %s",
                len(all_clauses),
                all_clauses,
            )

        return all_clauses

    def parse(
        self,
        text: str,
        context_subject: str | None = None,
    ) -> list[InterpretData] | None:
        """
        Attempts to parse a sentence by running it through a multi-stage pipeline.
        Returns a list of interpretations (one for each successfully parsed clause).
        """
        logger.debug(f"  [Symbolic Parser]: Attempting to parse sentence: '{text}'")

        clean_text = text.lower().strip().rstrip("?")

        if clean_text in ("who are you", "what are you"):
            logger.info(
                "  [Symbolic Parser]: Successfully parsed a 'who are you' meta-question.",
            )
            return [
                {
                    "intent": "meta_question_self",
                    "entities": [{"name": "agent", "type": "CONCEPT"}],
                    "relation": None,
                    "key_topics": [],
                    "full_text_rephrased": "",
                },
            ]

        if clean_text in ("what is your purpose", "what's your purpose"):
            logger.info(
                "  [Symbolic Parser]: Successfully parsed a 'purpose' meta-question.",
            )
            return [
                {
                    "intent": "meta_question_purpose",
                    "entities": [{"name": "agent", "type": "CONCEPT"}],
                    "relation": None,
                    "key_topics": [],
                    "full_text_rephrased": "",
                },
            ]

        if clean_text in ("what can you do", "what are your abilities"):
            logger.info(
                "  [Symbolic Parser]: Successfully parsed an 'abilities' meta-question.",
            )
            return [
                {
                    "intent": "meta_question_abilities",
                    "entities": [{"name": "agent", "type": "CONCEPT"}],
                    "relation": None,
                    "key_topics": [],
                    "full_text_rephrased": "",
                },
            ]

        if clean_text == "show all facts":
            logger.info(
                "  [Symbolic Parser]: Successfully parsed 'show all facts' command.",
            )
            return [
                {
                    "intent": "command_show_all_facts",
                    "entities": [],
                    "relation": None,
                    "key_topics": ["show all facts"],
                    "full_text_rephrased": "User has issued a command to show all facts.",
                },
            ]

        clauses = self._split_into_clauses(text)
        all_interpretations: list[InterpretData] = []

        for clause in clauses:
            interpretation = self._parse_single_clause(clause, context_subject)
            if interpretation:
                all_interpretations.append(interpretation)

        if context_subject:
            logger.debug(
                f"  [Introspection Parse]: Context='{context_subject}', text='{text}'",
            )

        if all_interpretations:
            return all_interpretations

        logger.debug("  [Symbolic Parser]: Failed. No clauses could be parsed.")
        return None

    def _parse_single_clause(
        self,
        clause: str,
        context_subject: str | None = None,
    ) -> InterpretData | None:
        """Applies a prioritized sequence of rules to deconstruct a single clause."""

        if clause.lower() == "show all facts":
            logger.info(
                "  [Symbolic Parser]: Successfully parsed 'show all facts' command.",
            )
            return InterpretData(
                intent="command",
                entities=[],
                relation=None,
                key_topics=["show all facts"],
                full_text_rephrased="User issued a command to show all facts.",
            )

        if context_subject:
            clause_before = clause
            clause = re.sub(
                r"\b(it|they|them)\b",
                context_subject,
                clause,
                flags=re.IGNORECASE,
            )
            clause = re.sub(
                r"\b(its|their)\b",
                f"{context_subject}'s",
                clause,
                flags=re.IGNORECASE,
            )
            if clause != clause_before:
                logger.debug(
                    f"  [Symbolic Parser]: Resolved pronouns using context '{context_subject}': "
                    f"'{clause_before}' → '{clause}'",
                )

        raw_clause = clause.lower()
        raw_words = raw_clause.split()

        if not raw_words:
            return None

        if raw_words[0] in self.QUESTION_WORDS:
            logger.info("  [Symbolic Parser]: Successfully parsed a wh-question.")
            entity_name = (
                " ".join(raw_words[2:])
                if len(raw_words) > 2
                else " ".join(raw_words[1:])
            )
            entity_name = entity_name.replace("?", "").strip()
            entity_name = re.sub(r"^(is|are|was|were)\s+", "", entity_name)
            return InterpretData(
                intent="question_about_entity",
                entities=[{"name": entity_name, "type": "CONCEPT"}],
                relation=None,
                key_topics=[entity_name],
                full_text_rephrased=raw_clause,
            )

        capital_match = self.CAPITAL_PATTERN.match(raw_clause)
        if capital_match:
            groups = capital_match.groupdict()
            city = self.agent._clean_phrase(groups["city"])
            country = self.agent._clean_phrase(groups["country"])
            logger.info(
                f"  [Symbolic Parser]: Successfully parsed Capital structure: '{country}' -> 'has_capital' -> '{city}'.",
            )
            relation = RelationData(subject=country, verb="has_capital", object=city)
            return InterpretData(
                intent="statement_of_fact",
                entities=[
                    {"name": city, "type": "CONCEPT"},
                    {"name": country, "type": "CONCEPT"},
                ],
                relation=relation,
                key_topics=[city, country],
                full_text_rephrased=raw_clause,
            )

        preposition_match = self.PREPOSITION_PATTERN.match(raw_clause)
        if preposition_match:
            groups = preposition_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb = groups["verb"].lower()
            preposition = groups["preposition"].lower()
            prep_object = self.agent._clean_phrase(groups["prep_object"])
            relation_type = self.PREPOSITION_TO_RELATION_MAP.get((verb, preposition))
            if relation_type:
                logger.info(
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
                    ],
                    relation=relation,
                    key_topics=[subject, prep_object],
                    full_text_rephrased=raw_clause,
                )

        svo_match = self.SVO_PATTERN.match(raw_clause)
        if svo_match:
            groups = svo_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb_raw = groups["verb"]
            object_ = self.agent._clean_phrase(groups["object"])
            verb = self.agent.get_relation_type(verb_raw, subject, object_)

            logger.info(
                f"  [Symbolic Parser]: Successfully parsed S-V-O structure via regex: '{subject}' -> '{verb}' -> '{object_}'.",
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
                full_text_rephrased=raw_clause,
            )

        words = []
        for w in raw_words:
            if self._is_part_of_speech(w, "verb"):
                words.append(self.agent.lemmatizer.lemmatize(w, pos="v"))
            else:
                words.append(self.agent.lemmatizer.lemmatize(w))

        lemmatized_clause = " ".join(words)

        relational_match = self.RELATIONAL_QUESTION_PATTERN.match(lemmatized_clause)
        if relational_match:
            groups = relational_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb = groups["verb"].lower()
            object_ = self.agent._clean_phrase(groups["object"])
            action_object = f"{verb} {object_}"
            logger.info(
                "  [Symbolic Parser]: Successfully parsed Yes/No Question: '%s' --[has_property]--> '%s'?",
                subject,
                action_object,
            )
            relation = RelationData(
                subject=subject,
                verb="has_property",
                object=action_object,
            )
            return InterpretData(
                intent="question_yes_no",
                entities=[
                    {"name": subject, "type": "CONCEPT"},
                    {"name": action_object, "type": "PROPERTY"},
                ],
                relation=relation,
                key_topics=[subject, action_object],
                full_text_rephrased=lemmatized_clause,
            )

        if words[0] in self.QUESTION_WORDS:
            logger.info("  [Symbolic Parser]: Successfully parsed a wh-question.")
            entity_name = " ".join(words[2:]) if len(words) > 2 else " ".join(words[1:])
            entity_name = entity_name.replace("?", "").strip()
            entity_name = re.sub(r"^(is|are|was|were)\s+", "", entity_name)
            return InterpretData(
                intent="question_about_entity",
                entities=[{"name": entity_name, "type": "CONCEPT"}],
                relation=None,
                key_topics=[entity_name],
                full_text_rephrased=lemmatized_clause,
            )

        yes_no_match = self.YES_NO_ADJECTIVE_PATTERN.match(lemmatized_clause)
        if yes_no_match:
            groups = yes_no_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            adjective = self.agent._clean_phrase(groups["adjective"])
            logger.info(
                f"  [Symbolic Parser]: Successfully parsed Yes/No Adjective Question: '{subject}' --[has_property]--> '{adjective}'?",
            )
            relation = RelationData(
                subject=subject,
                verb="has_property",
                object=adjective,
            )
            return InterpretData(
                intent="question_yes_no",
                entities=[
                    {"name": subject, "type": "CONCEPT"},
                    {"name": adjective, "type": "PROPERTY"},
                ],
                relation=relation,
                key_topics=[subject, adjective],
                full_text_rephrased=lemmatized_clause,
            )

        property_of_match = re.match(
            r"(?i)^the\s+(?P<property>\w+)\s+of\s+(?P<subject>.+?)\s+is\s+(?P<value>.+)\.*$",
            lemmatized_clause,
        )
        if property_of_match:
            groups = property_of_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            property_name = self.agent._clean_phrase(groups["property"])
            value = self.agent._clean_phrase(groups["value"])
            logger.info(
                f"  [Symbolic Parser]: Successfully parsed 'Property of Subject' structure: '{subject}' has property '{value}'.",
            )
            relation = RelationData(subject=subject, verb="has_property", object=value)
            return InterpretData(
                intent="statement_of_fact",
                entities=[
                    {"name": subject, "type": "CONCEPT"},
                    {"name": value, "type": "PROPERTY"},
                ],
                relation=relation,
                key_topics=[subject, property_name, value],
                full_text_rephrased=lemmatized_clause,
            )

        verb_info = self._find_verb(words)
        if not verb_info:
            return None
        verb, verb_index = verb_info

        if len(words) > verb_index + 1:
            potential_adjective = words[verb_index + 1]
            if self._is_part_of_speech(potential_adjective, "adjective"):
                subject = " ".join(words[:verb_index])
                subject = self.agent._clean_phrase(subject)
                logger.info(
                    "  [Symbolic Parser]: Successfully parsed S-V-Adjective structure: '%s' has property '%s'.",
                    subject,
                    potential_adjective,
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
                    full_text_rephrased=lemmatized_clause,
                )

        if verb_index > 0 and verb_index < len(words) - 1:
            subject = " ".join(words[:verb_index])
            verb_phrase = words[verb_index]
            object_start_index = verb_index + 1
            if len(words) > verb_index + 1 and (
                self._is_part_of_speech(words[verb_index + 1], "verb")
                or self._is_part_of_speech(words[verb_index + 1], "preposition")
            ):
                verb_phrase += "_" + words[verb_index + 1]
                object_start_index += 1
            verb = verb_phrase
            object_ = " ".join(words[object_start_index:])
            subject = self.agent._clean_phrase(subject)
            object_ = self.agent._clean_phrase(object_)
            logger.info(
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
                full_text_rephrased=lemmatized_clause,
            )

        return None

    def _resolve_clause_pronouns(self, clause: str, subject: str) -> str:
        """Replaces pronouns in a clause with a given subject."""
        modified_clause = re.sub(
            r"\b(it|they|them|he|she)\b",
            subject,
            clause,
            flags=re.IGNORECASE,
        )
        modified_clause = re.sub(
            r"\b(its|their|his|her)\b",
            f"{subject}'s",
            modified_clause,
            flags=re.IGNORECASE,
        )
        if modified_clause != clause:
            logger.debug(
                f"  [Parser Coreference]: Resolved clause: '{clause}' -> '{modified_clause}'",
            )
        return modified_clause

    def _find_verb(self, words: list[str]) -> tuple[str, int] | None:
        """Scan a list of words to find the most likely single verb.

        This helper function iterates through the words of a sentence and
        uses the `LexiconManager` to check if any of them are categorized
        as a 'verb'.

        If multiple verbs are found (e.g., in a phrasal verb like "give
        birth"), it applies a simple heuristic and prioritizes the first
        one it finds.

        Args:
            words: A list of tokenized words from the input sentence.

        Returns:
            A tuple containing the verb string and its index, or None if
            no known verbs are found.
        """
        found_verbs = []
        for i, word in enumerate(words):
            if self._is_part_of_speech(word, "verb"):
                found_verbs.append((word, i))

        if not found_verbs:
            return None

        if len(found_verbs) == 1:
            return found_verbs[0]

        logger.debug(
            f"  [Parser]: Multiple verbs found: {found_verbs}. Prioritizing the first.",
        )
        return found_verbs[0]

    def _is_part_of_speech(self, word: str, pos: str) -> bool:
        """
        Check if a word is categorized as a specific part of speech.
        This is a read-only check against the agent's current knowledge.
        """
        clean_word = word.lower().strip()
        word_node = self.agent.graph.get_node_by_name(clean_word)
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
