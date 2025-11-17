from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# --- FIX: Remove NLTK, import spaCy ---
import spacy

from axiom.universal_interpreter import InterpretData, RelationData

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

    from axiom.cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class SymbolicParser:
    """
    A deterministic, dependency-parsing-first, rule-based parser using spaCy
    for robust grammatical analysis.
    """

    __slots__ = ("agent", "nlp", "QUESTION_WORDS")

    def __init__(self, agent: CognitiveAgent) -> None:
        """Initialize the SymbolicParser and load the spaCy model."""
        self.agent = agent
        self.QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which"}
        try:
            # Use the small model for speed and efficiency
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

    def parse(
        self, text: str, main_topic: str | None = None
    ) -> list[InterpretData] | None:
        """
        Attempts to parse a sentence by performing a full dependency parse with spaCy
        and then applying rules to the grammatical tree.
        """
        logger.debug(f"  [Symbolic Parser]: Parsing with spaCy: '{text}'")

        # spaCy does the heavy lifting of understanding the grammar
        doc: Doc = self.nlp(text)

        # Find the root of the sentence, which is almost always the main verb
        root_verb: Token | None = None
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root_verb = token
                break

        if not root_verb:
            logger.debug(
                "  [Symbolic Parser]: Failed. No root verb found in the sentence."
            )
            return None

        # --- Now, we can apply simple, reliable rules to the dependency tree ---
        return self._handle_svo_statement(doc, root_verb, main_topic)

    def _get_full_noun_phrase(self, token: Token) -> str:
        """Helper to extract the full noun phrase (e.g., 'the big red ball') from a token."""
        return " ".join(t.text for t in token.subtree)

    def _handle_svo_statement(
        self, doc: Doc, verb: Token, main_topic: str | None = None
    ) -> list[InterpretData] | None:
        """
        Extracts a Subject-Verb-Object relationship by walking the dependency tree
        from the root verb.
        """
        subjects = [
            self._get_full_noun_phrase(t)
            for t in verb.lefts
            if t.dep_ in ("nsubj", "nsubjpass")
        ]
        objects = [
            self._get_full_noun_phrase(t)
            for t in verb.rights
            if t.dep_ in ("dobj", "pobj", "attr")
        ]

        subject_str = subjects[0] if subjects else ""
        object_str = objects[0] if objects else ""

        # Use context if the subject is missing (e.g., for definitions)
        if not subject_str and main_topic:
            subject_str = main_topic
            logger.debug(
                f"  [Symbolic Parser]: Inferred subject '{main_topic}' from context."
            )

        if not subject_str or not object_str:
            logger.debug(
                "  [Symbolic Parser]: Failed to find a clear Subject or Object."
            )
            return None

        clean_subject = self.agent._clean_phrase(subject_str)
        clean_obj = self.agent._clean_phrase(object_str)

        if not clean_subject or not clean_obj:
            return None

        relation_verb = self.agent.get_relation_type(
            verb.lemma_, clean_subject, clean_obj
        )
        relation = RelationData(
            subject=clean_subject, verb=relation_verb, object=clean_obj
        )

        interp = InterpretData(
            intent="statement_of_fact",
            entities=[
                {"name": clean_subject, "type": "CONCEPT"},
                {"name": clean_obj, "type": "CONCEPT"},  # Simplified for now
            ],
            relation=relation,
            key_topics=[clean_subject, clean_obj],
            full_text_rephrased=doc.text,
            provenance="symbolic_parser",
            confidence=0.98,  # Higher confidence because it's based on a full parse
        )
        return [interp]
