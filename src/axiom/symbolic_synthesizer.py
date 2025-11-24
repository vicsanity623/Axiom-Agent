import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent
    from .graph_core import RelationshipEdge

logger = logging.getLogger(__name__)


class SymbolicSynthesizer:
    """
    A deterministic, template-based natural language generator.
    It converts structured graph data (RelationshipEdges) into human-readable text
    without relying on an LLM.
    """

    def __init__(self, agent: "CognitiveAgent"):
        self.agent = agent
        self._load_templates()

    def _load_templates(self):
        """Initializes the template library."""
        self.templates = {
            "is_a": [
                "{subject} is a type of {object}.",
                "{subject} is a {object}.",
                "{subject} falls under the category of {object}.",
            ],
            "has_property": [
                "{subject} is {object}.",
                "{subject} has the quality of being {object}.",
            ],
            "part_of": [
                "{subject} is part of {object}.",
                "{subject} is a component of {object}.",
            ],
            "capable_of": [
                "{subject} can {object}.",
                "{subject} is capable of {object}.",
            ],
            "used_for": [
                "{subject} is used for {object}.",
                "{subject} is used to {object}.",
            ],
            "located_in": [
                "{subject} is located in {object}.",
                "{subject} can be found in {object}.",
            ],
            "has_part": [
                "{subject} has {object}.",
                "{subject} contains {object}.",
            ],
            "has_name": [
                "The name is {object}.",
                "It is called {object}.",
            ],
            # Default fallback
            "default": [
                "{subject} {verb} {object}.",
            ],
        }

    def synthesize(
        self, facts: list[tuple[str, "RelationshipEdge"]], topic: str
    ) -> str | None:
        """
        Synthesizes a natural language response from a list of facts.
        Returns None if synthesis fails or if the facts are too complex for templates.
        """
        if not facts:
            return None

        logger.info(
            f"[Synthesizer]: Attempting to synthesize response for '{topic}' with {len(facts)} facts."
        )

        sentences = []

        # Sort facts by weight/relevance if possible (assuming they come sorted or we sort here)
        # For now, we take the top 3-5 to avoid overwhelming lists
        top_facts = facts[:5]

        for fact_str, edge in top_facts:
            source_node = self.agent.graph.get_node_by_id(edge.source)
            target_node = self.agent.graph.get_node_by_id(edge.target)

            if not source_node or not target_node:
                continue

            # Determine subject and object names
            # If the source is the topic, use "It" or the name for the first sentence
            subject_text = source_node.name
            if source_node.name.lower() == topic.lower():
                # Use pronoun "It" for subsequent sentences if we implement state tracking
                # For now, keep it simple: use the name
                pass

            verb = edge.type
            object_text = target_node.name

            # Select template
            template_list = self.templates.get(verb, self.templates["default"])
            template = random.choice(template_list)

            # Format sentence
            try:
                sentence = template.format(
                    subject=subject_text,
                    verb=verb.replace("_", " "),
                    object=object_text,
                )
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:]
                sentences.append(sentence)
            except Exception as e:
                logger.warning(
                    f"[Synthesizer]: Failed to format template for {verb}: {e}"
                )
                continue

        if not sentences:
            return None

        # Join sentences into a paragraph
        return " ".join(sentences)
