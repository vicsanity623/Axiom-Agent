from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import spacy

from axiom.universal_interpreter import InterpretData, RelationData

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

    from axiom.cognitive_agent import CognitiveAgent

logger = logging.getLogger("axiom")


class SymbolicParser:
    """
    A deterministic, hybrid, rule-based parser using spaCy for robust grammatical analysis.
    It first checks for high-priority commands and specific patterns before falling back
    to a full dependency parse.
    """

    __slots__ = (
        "agent",
        "nlp",
        "QUESTION_WORDS",
        "META_NAME_PATTERN",
        "PREPOSITION_PATTERN",
        "PREPOSITION_TO_RELATION_MAP",
    )

    def __init__(self, agent: CognitiveAgent) -> None:
        self.agent = agent
        self.QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which"}
        self.META_NAME_PATTERN = re.compile(
            r"\b(what is|what'?s) your name\b", re.IGNORECASE
        )
        self.PREPOSITION_PATTERN = re.compile(
            r"(?i)^(?P<subject>.+?)\s+(?P<verb>is|are|was|were)\s+(?:.+?\s+)?(?P<preposition>in|on|at|of|from|with|inside)\s+(?P<prep_object>.+?)\.*$"
        )
        self.PREPOSITION_TO_RELATION_MAP = {
            ("is", "in"): "is_located_in",
            ("are", "in"): "is_located_in",
            ("is", "on"): "is_on_top_of",
            ("are", "on"): "is_on_top_of",
            ("is", "at"): "is_located_at",
            ("are", "at"): "is_located_at",
        }

        try:
            self.nlp: Language = spacy.load("en_core_web_sm")
            logger.info("[success]spaCy NLP pipeline loaded successfully.[/success]")
        except OSError:
            logger.error(
                "[bold red]CRITICAL ERROR: spaCy model 'en_core_web_sm' not found.[/bold red]"
            )
            logger.error(
                "[bold yellow]Please run 'uv sync --extra dev' to install dependencies.[/bold yellow]"
            )
            raise SystemExit("Halting due to missing spaCy model.")
        logger.info("[success]Symbolic Parser initialized.[/success]")

    def parse(self, text: str) -> list[InterpretData] | None:
        """
        Attempts to parse a sentence using a hybrid, rule-based approach.
        """
        logger.debug(f"  [Symbolic Parser]: Parsing with hybrid rules: '{text}'")
        clean_text = text.lower().strip().rstrip("?.,!")
        logger.debug(f"[Symbolic Parser]: Normalized text: '{clean_text}'")
        words = clean_text.split()

        # --- Your existing, correct fast-path rules ---
        if clean_text == "show all facts":
            return [
                InterpretData(
                    intent="command_show_all_facts",
                    entities=[],
                    relation=None,
                    key_topics=["show all facts"],
                    full_text_rephrased="User issued a command to show all facts.",
                    provenance="symbolic_parser_command_rule",
                    confidence=1.0,
                )
            ]
        if self.META_NAME_PATTERN.search(clean_text):
            return [
                InterpretData(
                    intent="meta_question_self",
                    entities=[{"name": "agent", "type": "CONCEPT"}],
                    relation=None,
                    key_topics=["agent", "name"],
                    full_text_rephrased=text,
                    provenance="symbolic_parser_meta_rule",
                    confidence=1.0,
                )
            ]
        if clean_text in ("who are you", "what are you"):
            return [
                InterpretData(
                    intent="meta_question_self",
                    entities=[{"name": "agent", "type": "CONCEPT"}],
                    relation=None,
                    key_topics=["agent"],
                    full_text_rephrased=text,
                    provenance="symbolic_parser_meta_rule",
                    confidence=1.0,
                )
            ]
        if clean_text in ("what is your purpose", "what's your purpose"):
            return [
                InterpretData(
                    intent="meta_question_purpose",
                    entities=[{"name": "agent", "type": "CONCEPT"}],
                    relation=None,
                    key_topics=["agent", "purpose"],
                    full_text_rephrased=text,
                    provenance="symbolic_parser_meta_rule",
                    confidence=1.0,
                )
            ]
        if clean_text in ("what can you do", "what are your abilities"):
            return [
                InterpretData(
                    intent="meta_question_abilities",
                    entities=[{"name": "agent", "type": "CONCEPT"}],
                    relation=None,
                    key_topics=["agent", "abilities"],
                    full_text_rephrased=text,
                    provenance="symbolic_parser_meta_rule",
                    confidence=1.0,
                )
            ]
        if (
            len(words) > 2
            and words[0] in self.QUESTION_WORDS
            and words[1] in ("is", "are")
        ):
            topic = " ".join(words[2:])
            clean_topic = self.agent._clean_phrase(topic)
            if clean_topic:
                return [
                    InterpretData(
                        intent="question_about_concept",
                        entities=[{"name": clean_topic, "type": "CONCEPT"}],
                        relation=None,
                        key_topics=[clean_topic],
                        full_text_rephrased=text,
                        provenance="symbolic_parser_wh_question_rule",
                        confidence=1.0,
                    )
                ]

        preposition_match = self.PREPOSITION_PATTERN.match(clean_text)
        if preposition_match:
            groups = preposition_match.groupdict()
            relation_type = self.PREPOSITION_TO_RELATION_MAP.get(
                (groups["verb"], groups["preposition"])
            )
            if relation_type:
                subject = self.agent._clean_phrase(groups["subject"])
                prep_object = self.agent._clean_phrase(groups["prep_object"])
                relation = RelationData(
                    subject=subject, verb=relation_type, object=prep_object
                )
                return [
                    InterpretData(
                        intent="statement_of_fact",
                        entities=[
                            {"name": subject, "type": "CONCEPT"},
                            {"name": prep_object, "type": "CONCEPT"},
                        ],
                        relation=relation,
                        key_topics=[subject, prep_object],
                        full_text_rephrased=text,
                        provenance="symbolic_parser_preposition_rule",
                        confidence=1.0,
                    )
                ]

        # --- FIX: The old, buggy regex for relational questions is GONE. ---
        # We will now handle this case with the more robust spaCy parser below.

        # --- Fallback: Full dependency parse for ALL other sentences ---
        logger.debug(
            "  [Symbolic Parser]: No specific rules matched. Performing full dependency parse."
        )
        doc: Doc = self.nlp(text)
        self._debug_dependency_tree(doc)

        if not doc:
            return None

        # --- FIX: Use a simpler, more robust routing logic ---
        # Rule 1: If the sentence starts with an auxiliary verb, it's a yes/no question.
        if doc[0].pos_ == "AUX":
            return self._handle_yes_no_question(doc)

        # Rule 2: If the root of the sentence is a verb, it's likely a statement.
        root: Token | None = next(
            (token for token in doc if token.dep_ == "ROOT"), None
        )
        if root and root.pos_ == "VERB":
            return self._handle_svo_statement(doc, root)

        logger.debug(
            "  [Symbolic Parser]: Failed. No spaCy handler matched the sentence structure."
        )
        return None

    def _debug_dependency_tree(self, doc: Doc) -> None:
        """
        Logs each token in the dependency tree for debugging.
        """
        logger.debug("── Dependency Parse ──")
        for token in doc:
            logger.debug(
                f"Token='{token.text:15}' "
                f"POS='{token.pos_:6}' "
                f"DEP='{token.dep_:10}' "
                f"HEAD='{token.head.text}'"
            )
        logger.debug("── End Dependency Parse ──")

    def _get_full_noun_phrase(self, token: Token) -> str:
        """
        Helper to extract the full noun phrase by walking specific dependency links.
        Now includes:
        - determiners, adjectives, and compounds on the left
        - compounds, appositions, attributes, and adjectival complements on the right
        """
        # Left modifiers
        parts = [
            child.text
            for child in token.lefts
            if child.dep_ in ("det", "amod", "compound")
        ]

        # Head token itself
        parts.append(token.text)

        # Right modifiers
        parts.extend(
            child.text
            for child in token.rights
            if child.dep_ in ("compound", "appos", "attr", "acomp")
        )

        return " ".join(parts)

    def _handle_yes_no_question(self, doc: Doc) -> list[InterpretData] | None:
        """
        Correct handling for yes/no questions like 'is a cat an animal?'
        Only relies on spaCy noun chunks.
        """
        chunks = list(doc.noun_chunks)

        # Expect exactly two noun chunks: subject and object
        if len(chunks) == 2:
            subj_text = chunks[0].text
            obj_text = chunks[1].text

            clean_subject = self.agent._clean_phrase(subj_text)
            clean_object = self.agent._clean_phrase(obj_text)

            if not clean_subject or not clean_object:
                return None

            relation = RelationData(
                subject=clean_subject,
                verb="is_a",
                object=clean_object,
            )

            return [
                InterpretData(
                    intent="question_yes_no",
                    entities=[
                        {"name": clean_subject, "type": "CONCEPT"},
                        {"name": clean_object, "type": "CONCEPT"},
                    ],
                    relation=relation,
                    key_topics=[clean_subject, clean_object],
                    full_text_rephrased=doc.text,
                    provenance="symbolic_parser_yes_no_chunk_rule",
                    confidence=1.0,
                )
            ]

        return None

    def _handle_svo_statement(
        self, doc: Doc, verb: Token
    ) -> list[InterpretData] | None:
        logger.debug(
            f"[SVO] Handling SVO statement for verb '{verb.text}' ({verb.dep_})"
        )
        self._debug_dependency_tree(doc)

        subjects = [
            self._get_full_noun_phrase(t)
            for t in verb.children
            if t.dep_ in ("nsubj", "nsubjpass")
        ]
        objects = [
            self._get_full_noun_phrase(t)
            for t in verb.children
            if t.dep_ in ("dobj", "pobj", "attr")
        ]

        logger.debug(f"[SVO] Raw subjects: {subjects}")
        logger.debug(f"[SVO] Raw objects: {objects}")

        # AUX with adjectival complement — “The cat is hungry”
        if not objects and verb.pos_ == "AUX":
            for child in verb.children:
                if child.dep_ == "acomp":
                    objects = [self._get_full_noun_phrase(child)]
                    logger.debug(f"[SVO] Found adjectival complement: {objects}")
                    break

        subject_str = subjects[0] if subjects else ""
        object_str = objects[0] if objects else ""

        if not subject_str or not object_str:
            logger.debug("[SVO] Missing subject or object → cannot extract SVO.")
            return None

        logger.debug(f"[SVO] Raw subject phrase: '{subject_str}'")
        logger.debug(f"[SVO] Raw object phrase: '{object_str}'")

        clean_subject = self.agent._clean_phrase(subject_str)
        clean_obj = self.agent._clean_phrase(object_str)

        logger.debug(f"[SVO] Cleaned subject: '{clean_subject}'")
        logger.debug(f"[SVO] Cleaned object: '{clean_obj}'")

        if not clean_subject or not clean_obj:
            logger.debug("[SVO] Cleaning removed subject/object → dropping.")
            return None

        relation_verb = self.agent.get_relation_type(
            verb.text, clean_subject, clean_obj
        )
        logger.debug(f"[SVO] Final relation verb: '{relation_verb}'")
        logger.debug(
            f"[SVO] Final relation: {clean_subject}  {relation_verb}  {clean_obj}"
        )

        relation = RelationData(
            subject=clean_subject, verb=relation_verb, object=clean_obj
        )

        return [
            InterpretData(
                intent="statement_of_fact",
                entities=[
                    {"name": clean_subject, "type": "CONCEPT"},
                    {"name": clean_obj, "type": "CONCEPT"},
                ],
                relation=relation,
                key_topics=[clean_subject, clean_obj],
                full_text_rephrased=doc.text,
                provenance="symbolic_parser",
                confidence=0.98,
            )
        ]
