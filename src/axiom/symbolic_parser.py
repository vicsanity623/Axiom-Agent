from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
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

        # Use a local variable to hold the loaded model first.
        # We assert it is of type Language (not Optional) because the except block handles failure.
        loaded_nlp: Language

        try:
            if getattr(sys, "frozen", False):
                if hasattr(sys, "_MEIPASS"):
                    # Onefile mode
                    bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
                    model_root = bundle_dir / "en_core_web_sm"
                else:
                    # Onedir mode (.app) - Model is in Contents/Frameworks
                    # sys.executable is .../Contents/MacOS/Axiom Studio
                    bundle_dir = Path(sys.executable).parent.parent / "Frameworks"
                    model_root = bundle_dir / "en_core_web_sm"

                # The model files are likely inside a subdirectory like 'en_core_web_sm-3.x.x'
                if model_root.exists():
                    subdirs = [
                        d
                        for d in model_root.iterdir()
                        if d.is_dir() and d.name.startswith("en_core_web_sm")
                    ]
                    if subdirs:
                        model_path = subdirs[0]
                    else:
                        model_path = model_root
                else:
                    model_path = model_root

                loaded_nlp = spacy.load(model_path)
            else:
                # FIX: Use underscores 'en_core_web_sm', not dashes
                loaded_nlp = spacy.load("en_core_web_sm")

            logger.info(
                "[success]Symbolic Parser's spaCy NLP pipeline loaded successfully.[/success]"
            )
        except Exception as e:
            logger.error(
                "[bold red]CRITICAL ERROR: Symbolic Parser failed to load spaCy model.[/bold red]"
            )
            logger.error("Original error: %s", e)
            raise SystemExit("Halting due to missing spaCy model in SymbolicParser.")

        # Assign to the instance attribute exactly once, satisfying mypy
        self.nlp = loaded_nlp
        logger.info("[success]Symbolic Parser initialized.[/success]")

    def parse(self, text: str) -> list[InterpretData] | None:
        """
        Attempts to parse a sentence using a hybrid, rule-based approach.
        """
        logger.debug(f"  [Symbolic Parser]: Parsing with hybrid rules: '{text}'")
        clean_text = text.lower().strip().rstrip("?.,!")
        words = clean_text.split()

        # --- Rule 0: High-priority, non-grammatical commands ---
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

        # --- Rule 1: High-priority, specific meta-question patterns ---
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

        # --- FIX: Rule 1.5: Add a "Question Gate" for simple "what is X?" questions ---
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

        # --- Rule 2: Specific grammatical patterns ---
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

        # --- Fallback: Rule 3: Full dependency parse for general statements ---
        logger.debug(
            "  [Symbolic Parser]: No specific rules matched. Performing full dependency parse."
        )
        doc: Doc = self.nlp(text)

        if not doc:
            return None

        # --- FIX: RE-INSERTED DEBUG LOOP HERE ---
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("  ── Dependency Parse ──")
            for token in doc:
                logger.debug(
                    f"  Token='{token.text:<15}' POS='{token.pos_:<6}' "
                    f"DEP='{token.dep_:<10}' HEAD='{token.head.text}'"
                )
            logger.debug("  ── End Dependency Parse ──")
        # -----------------------------------------

        # --- FIX: Use a simpler, more robust routing logic ---
        # Rule 1: If the sentence starts with an auxiliary verb, it's a yes/no question.
        if doc[0].pos_ == "AUX":
            return self._handle_yes_no_question(doc)

        # Rule 2: If the root of the sentence is a verb, it's likely a statement.
        root: Token | None = next(
            (token for token in doc if token.dep_ == "ROOT"), None
        )

        if root:
            if root.pos_ == "VERB":
                return self._handle_svo_statement(doc, root)
            # --- FIX: Handle Copular Verbs (is, are, was, were) where POS is AUX ---
            if root.pos_ == "AUX":
                return self._handle_copular_statement(doc, root)

        logger.debug(
            f"  [Symbolic Parser]: Failed. No spaCy handler matched for root '{root.text if root else 'None'}' ({root.pos_ if root else 'None'})."
        )
        return None

    def _handle_copular_statement(
        self, doc: Doc, root: Token
    ) -> list[InterpretData] | None:
        """
        Handles sentences where the main verb is a copula (is, are, was, were),
        linking a subject to an attribute (predicate nominative/adjective).
        Example: "A kitchen is a room." -> (kitchen, is_a, room)
        """
        # 1. Find Subject (nsubj)
        subj = next((c for c in root.children if c.dep_ == "nsubj"), None)
        if not subj:
            return None

        # 2. Find Attribute (attr), Adjectival Complement (acomp), or Open Clausal Complement (xcomp)
        # "Kitchen is a room" -> room (attr)
        # "The sky is blue" -> blue (acomp)
        # "Reasoning is to deduce" -> deduce (xcomp)
        attr = next(
            (c for c in root.children if c.dep_ in ("attr", "acomp", "xcomp")), None
        )

        # If no direct attribute, check for prepositional phrases acting as predicates?
        # e.g. "The cat is on the mat" -> on (prep)
        if not attr:
            attr = next((c for c in root.children if c.dep_ == "prep"), None)

        if not attr:
            return None

        subject_str = self._get_full_noun_phrase(subj)

        # For attributes, we might want the full phrase too
        if attr.dep_ == "prep":
            # "on the mat"
            # This is getting complex, maybe just take the pobj of the prep?
            pobj = next((c for c in attr.children if c.dep_ == "pobj"), None)
            if pobj:
                object_str = f"{attr.text} {self._get_full_noun_phrase(pobj)}"
                relation_type = (
                    "is_located_in"  # Defaulting to location for preps? Or generic?
                )
                # Let's use the map if possible, or just "is_related_to"
                if attr.text in ("in", "on", "at"):
                    relation_type = f"is_located_{attr.text}"
                else:
                    relation_type = "is_related_to"
            else:
                return None
        elif attr.dep_ == "xcomp":
            # "to deduce..." -> treat as "means" or "involves"
            # We need to capture the full verb phrase "to deduce..."
            # Simple approach: get the subtree? Or just the verb?
            # Let's try to reconstruct "to [verb] ..."
            # Check for 'aux' child 'to'
            aux_to = next(
                (
                    c
                    for c in attr.children
                    if c.dep_ == "aux" and c.text.lower() == "to"
                ),
                None,
            )
            if aux_to:
                object_str = f"to {self._get_full_noun_phrase(attr)}"  # _get_full_noun_phrase might not be perfect for verbs but let's try
                # Actually, for verbs we want the full subtree usually.
                # Let's just use the token text for now to be safe, or a simple reconstruction
                object_str = f"to {attr.text}"
                # If there are objects of the xcomp, add them? e.g. "to deduce a conclusion"
                dobj = next((c for c in attr.children if c.dep_ == "dobj"), None)
                if dobj:
                    object_str += f" {self._get_full_noun_phrase(dobj)}"

                relation_type = "involves_action"  # Or "means"
            else:
                object_str = self._get_full_noun_phrase(attr)
                relation_type = "involves"
        else:
            object_str = self._get_full_noun_phrase(attr)
            relation_type = "is_a" if attr.dep_ == "attr" else "has_property"

        clean_subject = self.agent._clean_phrase(subject_str)
        clean_obj = self.agent._clean_phrase(object_str)

        if not clean_subject or not clean_obj:
            return None

        relation = RelationData(
            subject=clean_subject, verb=relation_type, object=clean_obj
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
                provenance="symbolic_parser_copula",
                confidence=0.95,
            )
        ]

    def _get_full_noun_phrase(self, token: Token) -> str:
        """
        Helper to extract the full noun phrase by walking specific dependency links.
        """
        parts = [
            child.text
            for child in token.lefts
            if child.dep_ in ("det", "amod", "compound")
        ]
        parts.append(token.text)
        parts.extend(child.text for child in token.rights if child.dep_ in ("compound"))
        return " ".join(parts)

    def _handle_yes_no_question(self, doc: Doc) -> list[InterpretData] | None:
        """
        Robust handler for yes/no questions.
        """
        # Find an AUX that governs subject/object relationships
        aux = next((t for t in doc if t.dep_ == "aux" and t.head.dep_ == "ROOT"), None)
        if not aux:
            # Fallback: look for sentence-initial or early AUX
            aux = next((t for t in doc if t.pos_ == "AUX"), None)

        if not aux:
            return None

        subj = next((c for c in aux.children if c.dep_ == "nsubj"), None)
        obj = next(
            (c for c in aux.children if c.dep_ in ("attr", "dobj", "oprd")), None
        )

        if not subj or not obj:
            # The AI's fallback to aux.head is a good idea if the direct children fail.
            if aux.head:
                subj = next((c for c in aux.head.children if c.dep_ == "nsubj"), None)
                obj = next(
                    (
                        c
                        for c in aux.head.children
                        if c.dep_ in ("attr", "dobj", "oprd")
                    ),
                    None,
                )

        if not subj or not obj:
            logger.debug(
                "  [Symbolic Parser]: Could not determine subject/predicate for yes/no question."
            )
            return None

        subject_str = self._get_full_noun_phrase(subj)
        object_str = self._get_full_noun_phrase(obj)

        clean_subject = self.agent._clean_phrase(subject_str)
        clean_obj = self.agent._clean_phrase(object_str)

        if not clean_subject or not clean_obj:
            return None

        relation = RelationData(subject=clean_subject, verb="is_a", object=clean_obj)

        return [
            InterpretData(
                intent="question_yes_no",
                entities=[
                    {"name": clean_subject, "type": "CONCEPT"},
                    {"name": clean_obj, "type": "CONCEPT"},
                ],
                relation=relation,
                key_topics=[clean_subject, clean_obj],
                full_text_rephrased=doc.text,
                provenance="symbolic_parser_yes_no",
                confidence=1.0,
            )
        ]

    def _handle_svo_statement(
        self, doc: Doc, verb: Token
    ) -> list[InterpretData] | None:
        """
        Extracts a Subject-Verb-Object relationship by walking the dependency tree.
        """
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

        if not objects and verb.pos_ == "AUX":
            for child in verb.children:
                if child.dep_ == "acomp":
                    objects = [self._get_full_noun_phrase(child)]
                    break

        subject_str = subjects[0] if subjects else ""
        object_str = objects[0] if objects else ""

        if not subject_str or not object_str:
            logger.debug(
                "  [Symbolic Parser]: Failed to find a clear Subject or Object in dependency tree."
            )
            return None

        clean_subject = self.agent._clean_phrase(subject_str)
        clean_obj = self.agent._clean_phrase(object_str)

        if not clean_subject or not clean_obj:
            return None

        relation_verb = self.agent.get_relation_type(
            verb.text, clean_subject, clean_obj
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
