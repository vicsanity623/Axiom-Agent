from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from axiom.universal_interpreter import InterpretData, RelationData

if TYPE_CHECKING:
    from spacy.language import Language

    from axiom.cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


try:
    import spacy

    nlp: Language | None = spacy.load("en-core-web-lg")
    logger.info("spaCy loaded successfully for POS tagging fallback.")
except ImportError:
    nlp = None
    logger.warning(
        "spaCy not found. POS tagging will rely solely on the agent's lexicon."
    )


class SymbolicParser:
    """A deterministic, rule-based parser for understanding simple language.

    This class is the core of the agent's native language understanding.
    It attempts to deconstruct a sentence into a structured `InterpretData`
    object by applying a series of grammatical rules. It relies on the
    agent's `LexiconManager` and a spaCy fallback to identify parts of speech.

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
        "PROPERTY_KEYWORDS",
        "AMBIGUOUS_PROPERTY_VERBS",
        "PROPERTY_OF_QUESTION_PATTERN",
        "PASSIVE_VOICE_PATTERN",
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
        self.PASSIVE_VOICE_PATTERN = re.compile(
            r"(?i)^(?P<object>.+?)\s+(is|are|was|were)\s+(?P<verb>\w+)\s+by\s+(?P<subject>.+?)\.*$",
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

        self.AMBIGUOUS_PROPERTY_VERBS = {"has", "have", "contains", "contain", "holds"}
        self.PROPERTY_KEYWORDS = {
            "shape",
            "color",
            "volume",
            "mass",
            "weight",
            "height",
            "length",
            "width",
            "temperature",
            "density",
            "state",
            "size",
        }

        self.PROPERTY_OF_QUESTION_PATTERN = re.compile(
            r"(?i)^what\s+(is|are)\s+(?:the\s+)?(?P<property>.+?)\s+of\s+(?P<subject>.+)\?*$",
        )

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
        Attempts to parse a sentence by running it through a multi-stage pipeline,
        including a refinement step to decompose complex objects.
        """
        logger.debug(f"  [Symbolic Parser]: Attempting to parse sentence: '{text}'")
        # --- CHANGED: Removed hardcoded meta-question checks. ---
        # This logic is now handled by the CognitiveAgent's pre-processing
        # and standard intent routing, which is more robust.
        clean_text = text.lower().strip().rstrip("?")
        if clean_text == "show all facts":
            return [
                InterpretData(
                    intent="command_show_all_facts",
                    entities=[],
                    relation=None,
                    key_topics=["show all facts"],
                    full_text_rephrased="User has issued a command to show all facts.",
                )
            ]
        initial_interpretations = []
        clauses = self._split_into_clauses(text)
        for clause in clauses:
            if interp := self._parse_single_clause(clause, context_subject):
                initial_interpretations.append(interp)
        if not initial_interpretations:
            logger.debug(
                "  [Symbolic Parser]: Failed. No initial clauses could be parsed.",
            )
            return None

        final_interpretations: list[InterpretData] = []

        def to_str(value: str | list[Any] | dict[str, Any] | None) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                name = value.get("name", "")
                return str(name) if name is not None else ""
            if isinstance(value, list):
                return " ".join(map(str, value))
            return ""

        for interp in initial_interpretations:
            final_interpretations.append(interp)
            relation = interp.get("relation")
            if relation:
                subject_str = to_str(relation.get("subject"))
                object_str = to_str(relation.get("object"))

                if subject_str and len(object_str.split()) > 2:
                    logger.debug(
                        "  [Parser Refinement]: Decomposing chunky object: '%s'",
                        object_str,
                    )
                    refined_relations = self._refine_object_phrase(
                        subject_str,
                        object_str,
                    )
                    for ref_rel in refined_relations:
                        ref_subject = to_str(ref_rel.get("subject"))
                        ref_object = to_str(ref_rel.get("object"))
                        ref_verb = to_str(ref_rel.get("verb"))
                        logger.info(
                            "  [Parser Refinement]: Extracted: %s -> %s -> %s",
                            ref_subject,
                            ref_verb,
                            ref_object,
                        )
                        new_interp = InterpretData(
                            intent="statement_of_fact",
                            relation=ref_rel,
                            entities=[
                                {"name": ref_subject, "type": "CONCEPT"},
                                {"name": ref_object, "type": "CONCEPT"},
                            ],
                            key_topics=[ref_subject, ref_object],
                            full_text_rephrased=f"{ref_subject} {ref_verb} {ref_object}",
                        )
                        final_interpretations.append(new_interp)
        return final_interpretations

    def _parse_single_clause(
        self,
        clause: str,
        context_subject: str | None = None,
    ) -> InterpretData | None:
        """Applies a prioritized sequence of rules to deconstruct a single clause."""
        if clause.lower() == "show all facts":
            return InterpretData(
                intent="command_show_all_facts",
                entities=[],
                relation=None,
                key_topics=["show all facts"],
                full_text_rephrased="User issued a command to show all facts.",
            )
        if context_subject:
            clause = self._resolve_clause_pronouns(clause, context_subject)
        raw_clause = clause.lower()
        raw_words = raw_clause.split()
        if not raw_words:
            return None
        prop_of_match = self.PROPERTY_OF_QUESTION_PATTERN.match(raw_clause)
        if prop_of_match:
            groups = prop_of_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            property_name = self.agent._clean_phrase(groups["property"])
            relation_verb = f"has_{property_name.replace(' ', '_')}"
            relation = RelationData(subject=subject, verb=relation_verb, object="?")
            return InterpretData(
                intent="question_by_relation",
                entities=[{"name": subject, "type": "CONCEPT"}],
                relation=relation,
                key_topics=[subject, property_name],
                full_text_rephrased=raw_clause,
            )
        if raw_words[0] in self.QUESTION_WORDS:
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
        passive_match = self.PASSIVE_VOICE_PATTERN.match(raw_clause)
        if passive_match:
            groups = passive_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb = self.agent._clean_phrase(groups["verb"])
            object_ = self.agent._clean_phrase(groups["object"])
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
        svo_match = self.SVO_PATTERN.match(raw_clause)
        if svo_match:
            groups = svo_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb_raw = groups["verb"]
            object_ = self.agent._clean_phrase(groups["object"])
            verb = self.agent.get_relation_type(verb_raw, subject, object_)
            if verb_raw.lower() in self.AMBIGUOUS_PROPERTY_VERBS:
                if any(kw in object_.lower() for kw in self.PROPERTY_KEYWORDS):
                    verb = "has_property"
            relation = RelationData(subject=subject, verb=verb, object=object_)
            return InterpretData(
                intent="statement_of_fact",
                entities=[
                    {"name": subject, "type": "CONCEPT"},
                    {
                        "name": object_,
                        "type": "PROPERTY" if verb == "has_property" else "CONCEPT",
                    },
                ],
                relation=relation,
                key_topics=[subject, object_],
                full_text_rephrased=raw_clause,
            )
        words = [self.agent.lemmatizer.lemmatize(w) for w in raw_words]
        lemmatized_clause = " ".join(words)
        relational_match = self.RELATIONAL_QUESTION_PATTERN.match(lemmatized_clause)
        if relational_match:
            groups = relational_match.groupdict()
            subject = self.agent._clean_phrase(groups["subject"])
            verb = groups["verb"].lower()
            object_ = self.agent._clean_phrase(groups["object"])
            action_object = f"{verb} {object_}"
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
        """Scan a list of words to find the most likely single verb."""
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

    @lru_cache(maxsize=1024)
    def _is_part_of_speech(self, word: str, pos: str) -> bool:
        """
        Check if a word is categorized as a specific part of speech.
        This is a read-only check against the agent's current knowledge.
        """
        clean_word = word.lower().strip()
        word_node = self.agent.graph.get_node_by_name(clean_word)
        if not word_node:
            if nlp:
                doc = nlp(clean_word)
                if doc and doc[0].pos_.lower() == pos:
                    return True
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

    def _refine_object_phrase(
        self,
        subject: str,
        object_phrase: str,
    ) -> list[RelationData]:
        """
        Analyzes a complex object phrase to extract additional atomic facts related
        to either the original subject or the nouns within the phrase itself.
        """
        refined_facts = []
        words = object_phrase.split()
        for i, word in enumerate(words):
            if self._is_part_of_speech(word, "adjective"):
                if i + 1 < len(words):
                    noun = self.agent._clean_phrase(words[i + 1])
                    refined_facts.append(
                        RelationData(subject=noun, verb="has_property", object=word),
                    )
        prep_match = re.search(r"(.+?)\s+(from|of|in|with)\s+(.+)", object_phrase)
        if prep_match:
            obj_subject = self.agent._clean_phrase(prep_match.group(1))
            preposition = prep_match.group(2)
            prep_object = self.agent._clean_phrase(prep_match.group(3))
            if preposition == "from":
                refined_facts.append(
                    RelationData(
                        subject=obj_subject,
                        verb="comes_from",
                        object=prep_object,
                    ),
                )
            elif preposition == "of":
                refined_facts.append(
                    RelationData(
                        subject=obj_subject,
                        verb="is_part_of",
                        object=prep_object,
                    ),
                )
            elif preposition == "with":
                refined_facts.append(
                    RelationData(
                        subject=obj_subject,
                        verb="has_part",
                        object=prep_object,
                    ),
                )
        return refined_facts
