from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE
from ..knowledge_harvester import KnowledgeHarvester
from ..logging_config import setup_logging
from ..metacognitive_engine import MetacognitiveEngine
from .cycle_manager import CycleManager

logger = logging.getLogger(__name__)


def start_autonomous_training(brain_file: Path, state_file: Path) -> None:
    """Initialize the agent and run its autonomous learning cycles indefinitely."""

    logger.info("--- [AUTONOMOUS TRAINER]: Starting Axiom Agent Initialization... ---")
    agent_interaction_lock = threading.Lock()

    try:
        axiom_agent = CognitiveAgent(
            brain_file=brain_file,
            state_file=state_file,
            inference_mode=False,
        )

        harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        metacognitive_engine = MetacognitiveEngine(
            agent=axiom_agent,
            gemini_api_key=gemini_api_key,
        )

        scheduler = BackgroundScheduler(daemon=True)
        manager = CycleManager(scheduler, harvester, metacognitive_engine)

        axiom_agent.goal_manager.add_goal(
            "Master linguistic communication through language ideology critique. "
            "[Stage 1: Ideology Recognition]: Identify implicit beliefs about language value and correctness. "
            "[Stage 2: Power Analysis]: Understand how language ideologies reinforce social hierarchies. "
            "[Stage 3: Standard Language Deconstruction]: Critique notions of linguistic purity and standardization. "
            "[Stage 4: Multilingual Advocacy]: Promote positive attitudes toward linguistic diversity. "
            "[Stage 5: Critical Language Awareness]: Develop reflective understanding of one's own language ideologies."
            "[Stage 6: Communication Masatery]: Achieve communication mastery through narrative competence development. "
            "[Stage 7: Event Sequencing]: Arrange events in temporally and causally coherent order. "
            "[Stage 8: Character Development]: Create and maintain consistent character identities and motivations. "
            "[Stage 9: Perspective Management]: Control narrative point of view and focalization. "
            "[Stage 10: Evaluative Framing]: Signal the significance and emotional valence of narrated events. "
            "[Stage 11: Genre Adaptation]: Tailor narrative strategies to different storytelling traditions and purposes."
            "[Stage 12: Pragmatics]: Study how context, intent, and social conventions contribute to meaning beyond the literal interpretation."
            "[Stage 13: Core Inquisition Goals]: Learn to communicate by following a developmental path, starting with simple labels and progressing to complex social dialogue. "
            "[Stage 14: Object Permanence and Labeling]: Learn to associate simple words (nouns) with concrete objects and concepts in the world. "
            "[Stage 15: Action and Property Association]: Learn to describe objects and events using simple verbs and adjectives. "
            "[Stage 16: Simple Sentence Formation]: Master the construction of basic Subject-Verb-Object sentences to state and request facts. "
            "[Stage 17: Theory of Mind]: Develop the ability to understand that others have intentions, goals, and perspectives different from one's own. "
            "[Stage 18: Collaborative Dialogue]: Practice using language to ask questions, resolve misunderstandings, and work with a user to achieve a shared goal."
            "[Stage 19: Core Inquisition Goals]: Grasp the essence of communication by exploring the connection between symbols, meaning, and reality. "
            "[Stage 20: The Symbol Grounding Problem]: Learn to connect abstract symbols (words) to their real-world referents and concepts. "
            "[Stage 21: Propositional Logic]: Master the ability to decompose sentences into logical propositions that can be evaluated for truth. "
            "[Stage 21: Semantic Truth Conditions]: Learn to determine the literal meaning of a statement by understanding the conditions under which it would be true. "
            "[Stage 22: Speech Act Theory]: Analyze how language is used not just to state facts, but to perform actions like promising, questioning, and commanding. "
            "[Stage 23: The Cooperative Principle]: Understand conversation as a cooperative game governed by implicit rules (Grice's Maxims) of quantity, quality, relation, and manner."
            "[Stage 24: Core Inquisition Goals]: Become an effective assistant by mastering the art of goal-oriented dialogue and task completion. "
            "[Stage 25: Core Vocabulary Acquisition]: Learn the essential nouns, verbs, and concepts related to common user tasks and domains. "
            "[Stage 26: Command and Query Parsing]: Develop a robust ability to parse user commands and questions into structured, actionable data. "
            "[Stage 27: Entity and Slot Recognition]: Master the identification of key pieces of information (entities) required to fulfill a request. "
            "[Stage 28: Disambiguation and Clarification]: Learn to detect ambiguity or missing information in a user’s request and proactively ask clarifying questions. "
            "[Stage 29: Multi-Turn Task Execution]: Practice maintaining the state of a complex task across multiple conversational turns to guide a user to a successful outcome."
        )

        manager.start()
        scheduler.start()

        logger.info("--- [AUTONOMOUS TRAINER]: Agent is running in headless mode. ---")
        logger.info("--- All learning cycles are active. Press CTRL+C to stop. ---")

        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        logger.info(
            "\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Exiting. ---",
        )
    except Exception as exc:
        logger.critical(
            "!!! [AUTONOMOUS TRAINER]: CRITICAL ERROR: %s !!!",
            exc,
            exc_info=True,
        )
        traceback.print_exc()
    finally:
        logger.info("--- [AUTONOMOUS TRAINER]: Process terminated. ---")


def main() -> None:
    """Entry point for the axiom-train command."""

    setup_logging()

    start_autonomous_training(DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE)


if __name__ == "__main__":
    main()
