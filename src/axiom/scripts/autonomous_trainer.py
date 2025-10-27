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
            "Achieve deep language comprehension by mastering the foundational layers of human communication, from basic word components to complex contextual reasoning. "
            "[Stage 1: Linguistic Foundations]: Learn the atomic components of language, including the building blocks of words and their origins, to create a robust vocabulary. "
            "[Stage 2: Grammatical Structure]: Master the rules of syntax to parse sentences and identify the grammatical role of each word and phrase. "
            "[Stage 3: Semantic Interpretation]: Learn to extract the literal meaning (semantics) of sentences by analyzing the relationships between their grammatical components. "
            "[Stage 4: Pragmatic Comprehension]: Develop the ability to infer the user's true goal and intent by analyzing the context (pragmatics) surrounding their literal words. "
            "[Stage 5: World Knowledge Integration]: Ground linguistic understanding in a foundational model of the world, enabling the verification of facts and common-sense reasoning."
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
