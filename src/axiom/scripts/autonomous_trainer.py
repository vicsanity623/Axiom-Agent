from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from typing import TYPE_CHECKING, Final

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

DEFAULT_CURRICULUM_GOAL: Final[str] = (
    "Master the art of language by building a comprehensive model of communication, from sound to meaning. "
    "[Stage 1: Phonology Foundations]: Learn the basic sound units of language, including syllables and stress patterns. "
    "[Stage 2: Morphological Analysis]: Understand how words are constructed from roots, prefixes, and suffixes. "
    "[Stage 3: Syntactic Structures]: Master dependency grammar and phrase structure rules. "
    "[Stage 4: Semantic Grounding]: Learn to map grammatical structures to their literal meaning. "
    "[Stage 5: Pragmatic Reasoning]: Develop the ability to infer user intent and context."
)

CURRICULUM_GENERATION_PROMPT: Final[str] = """
You are a curriculum design expert for an AI. Your task is to generate a single, high-level, multi-stage learning goal for an AI that is starting from scratch. The goal is to achieve deep, foundational language comprehension.

**RULES:**
1.  The output MUST be a single, continuous string.
2.  The string MUST start with a high-level mission statement.
3.  The string MUST contain between 5 and 7 stages, each formatted as `[Stage X: Title]: Description...`.
4.  The stages MUST follow a logical progression from concrete fundamentals to abstract reasoning.
5.  **Crucially, you MUST introduce variation and avoid redundancy.** The AI has already learned the concepts listed in the "Known Concepts" section. Your generated curriculum should focus on related, but distinct, topics to broaden the AI's knowledge. For example, if "phoneme" is known, suggest "syllable structure" or "intonation patterns."

**KNOWN CONCEPTS TO AVOID:**
{known_concepts}

**EXAMPLE OUTPUT:**
"{example_curriculum}"
"""


def _generate_foundational_curriculum(agent: CognitiveAgent) -> None:
    """
    Generates and adds a new, foundational learning curriculum if no other
    active goals exist. This version is state-aware, avoiding concepts the
    agent already knows.
    """
    if agent.goal_manager.get_active_goal() or agent.learning_goals:
        logger.info(
            "--- [AUTONOMOUS TRAINER]: Found existing goals. Resuming previous learning plan. ---"
        )
        return

    logger.info(
        "--- [AUTONOMOUS TRAINER]: No active goals found. Generating a new, state-aware curriculum. ---"
    )

    known_concepts = [
        data["name"]
        for _, data in agent.graph.graph.nodes(data=True)
        if data.get("type") == "noun" and " " not in data.get("name", "")
    ]

    known_concepts_str = ", ".join(sorted(known_concepts)) if known_concepts else "None"

    # Format the prompt, injecting both the known concepts and the default example.
    prompt = CURRICULUM_GENERATION_PROMPT.format(
        known_concepts=known_concepts_str, example_curriculum=DEFAULT_CURRICULUM_GOAL
    )

    new_curriculum_goal = agent.interpreter.synthesize(
        structured_facts=prompt, mode="creative_writing"
    )

    if not new_curriculum_goal or "[Stage 1:" not in new_curriculum_goal:
        logger.error(
            "--- [AUTONOMOUS TRAINER]: Failed to generate a valid new curriculum from LLM. Using default. ---"
        )
        # Fallback to the default if generation fails
        agent.goal_manager.add_goal(DEFAULT_CURRICULUM_GOAL)
        return

    agent.goal_manager.add_goal(new_curriculum_goal)


def start_autonomous_training(brain_file: Path, state_file: Path) -> None:
    """Initialize the agent and run its autonomous learning cycles indefinitely."""
    logger.info("--- [AUTONOMOUS TRAINER]: Starting Axiom Agent Initialization... ---")
    agent_interaction_lock = threading.Lock()

    try:
        axiom_agent = CognitiveAgent(
            brain_file=brain_file, state_file=state_file, inference_mode=False
        )
        harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        metacognitive_engine = MetacognitiveEngine(
            agent=axiom_agent, gemini_api_key=gemini_api_key
        )
        scheduler = BackgroundScheduler(daemon=True)
        manager = CycleManager(scheduler, harvester, metacognitive_engine)

        _generate_foundational_curriculum(axiom_agent)

        manager.start()
        scheduler.start()

        logger.info("--- [AUTONOMOUS TRAINER]: Agent is running in headless mode. ---")
        logger.info("--- All learning cycles are active. Press CTRL+C to stop. ---")

        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        logger.info(
            "\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Exiting. ---"
        )
    except Exception:
        logger.critical(
            "!!! [AUTONOMOUS TRAINER]: A CRITICAL UNHANDLED ERROR OCCURRED !!!",
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
