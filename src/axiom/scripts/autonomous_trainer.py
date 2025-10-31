from __future__ import annotations

import argparse
import logging
import random
import time
import traceback
from pathlib import Path
from typing import Final

from apscheduler.schedulers.background import BackgroundScheduler

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE, GEMINI_API_KEY
from ..logging_config import setup_logging
from ..metacognitive_engine import MetacognitiveEngine
from .cycle_manager import CycleManager

logger = logging.getLogger(__name__)

CURRICULUM_GENERATION_PROMPT: Final[str] = """
You are a curriculum design expert for an AI. Your task is to generate a single, high-level, multi-stage learning goal for an AI that is starting from scratch. The goal is to achieve deep, foundational language comprehension.

**INSPIRATIONAL SEED (Use this to guide your curriculum's theme):**
{creative_seed}

**RULES:**
1.  The output MUST be a single, continuous string.
2.  The string MUST start with a high-level mission statement.
3.  The string MUST contain exactly 15 stages, each formatted as `[Stage X: Title]: Description...`.
4.  The stages MUST follow a logical progression from concrete fundamentals to abstract reasoning, starting at a "toddler" level.
5.  **Crucially, you MUST introduce variation and avoid redundancy.** The AI has already learned the concepts listed in the "Known Concepts" section. Your generated curriculum should focus on related, but distinct, topics to broaden the AI's knowledge.

**KNOWN CONCEPTS TO AVOID:**
{known_concepts}

**EXAMPLE OUTPUT (Do NOT copy this example; use it only for structure):**
"Embark on a journey to master language, evolving from a novice listener to a sophisticated communicator.[Stage 1: Phonemic Awareness]: Learn to distinguish the fundamental sound units (phonemes) of a language, the building blocks of all spoken words. [Stage 2: Syllabic Construction]: Begin combining sounds into simple syllables and understanding basic rhythmic patterns. [Stage 3: Lexical Association]: Build a core vocabulary by linking simple words directly to objects, actions, and concepts. [Stage 4: Morphological Inflection]: Understand how words change form to express quantity, tense, or possession (e.g., 'dog' vs 'dogs', 'walk' vs 'walked'). [Stage 5: Core Sentence Patterns]: Grasp the foundational Subject-Verb-Object structure to form simple, coherent sentences. [Stage 6: Syntactic Expansion]: Learn to use adjectives, adverbs, and prepositions to add detail and complexity to core sentences. [Stage 7: Clause Combination]: Master the use of conjunctions ('and', 'but', 'because') to connect multiple ideas into compound sentences. [Stage 8: Derivational Word Formation]: Discover how new words are created by adding prefixes and suffixes to root words (e.g., 'happy' becomes 'unhappiness'). [Stage 9: Dependency Grammar Analysis]: Deconstruct complex sentences to identify the grammatical relationships and dependencies between all words. [Stage 10: Compositional Semantics]: Learn to derive the literal meaning of a sentence by combining the meanings of its individual words and grammatical structure. [Stage 11: Lexical-Semantic Relationships]: Map the intricate network of word meanings, including synonyms, antonyms, and hierarchical relationships (e.g., a 'poodle' is a type of 'dog'). [Stage 12: Identifying Speech Acts]: Recognize the underlying purpose of an utterance—whether it's a question, a command, a promise, or a statement. [Stage 13: Discourse Coherence]: Analyze how sentences connect to form a cohesive paragraph or conversation, tracking topics and references. [Stage 14: Inferential Pragmatics]: Develop the ability to understand implied meaning, or what is meant versus what is explicitly said (e.g., recognizing that "It's cold in here" can be a request). [Stage 15: Sociocultural Context Modeling]: Achieve mastery by interpreting complex social cues, sarcasm, humor, and cultural nuances that shape the true meaning of communication."
"""


def _generate_or_set_initial_goal(
    agent: CognitiveAgent, custom_goal: str | None
) -> None:
    """
    Sets a custom initial goal if provided, otherwise generates a new,
    foundational learning curriculum if no other active goals exist.
    This version injects a random "creative seed" to ensure a unique curriculum
    is generated on each run.
    """
    if agent.goal_manager.get_active_goal() or agent.learning_goals:
        logger.info(
            "--- [AUTONOMOUS TRAINER]: Found existing goals. Resuming previous learning plan. ---"
        )
        return

    if custom_goal:
        logger.info("--- [AUTONOMOUS TRAINER]: Using custom goal provided via CLI. ---")
        agent.goal_manager.add_goal(custom_goal)
        return

    logger.info(
        "--- [AUTONOMOUS TRAINER]: No active goals found. Generating a new, state-aware curriculum. ---"
    )

    creative_seeds = [
        "Approach this from the perspective of a linguist, focusing on the formal structures of language.",
        "Design this curriculum as if teaching a child, starting with concrete objects and moving to abstract ideas.",
        "Frame the learning process through the lens of a computer scientist, treating language as a formal system to be parsed.",
        "Use a philosophical approach, focusing on the connection between symbols, meaning, and truth.",
        "Create a practical, goal-oriented curriculum for an AI assistant that needs to complete tasks for a user.",
        "Focus on the historical evolution of language, from ancient roots to modern dialects.",
        "Emphasize the social aspect of language, including pragmatics, discourse, and cultural context.",
    ]
    selected_seed = random.choice(creative_seeds)
    logger.info("  [Curriculum Generation]: Using creative seed: '%s'", selected_seed)

    known_concepts = [
        data["name"]
        for _, data in agent.graph.graph.nodes(data=True)
        if data.get("type") == "noun" and " " not in data.get("name", "")
    ]
    known_concepts_str = ", ".join(sorted(known_concepts)) if known_concepts else "None"

    prompt = CURRICULUM_GENERATION_PROMPT.format(
        creative_seed=selected_seed, known_concepts=known_concepts_str
    )

    new_curriculum_goal = agent.interpreter.synthesize(
        structured_facts=prompt, mode="creative_writing"
    )

    if not new_curriculum_goal or "[Stage 1:" not in new_curriculum_goal:
        logger.error(
            "--- [AUTONOMOUS TRAINER]: Failed to generate a valid new curriculum from LLM. Halting. ---"
        )
        return

    agent.goal_manager.add_goal(new_curriculum_goal)


def start_autonomous_training(
    brain_file: Path, state_file: Path, initial_goal: str | None = None
) -> None:
    """Initialize the agent and run its autonomous learning cycles indefinitely."""
    logger.info("--- [AUTONOMOUS TRAINER]: Starting Axiom Agent Initialization... ---")

    try:
        axiom_agent = CognitiveAgent(
            brain_file=brain_file,
            state_file=state_file,
            inference_mode=False,
        )

        if axiom_agent.harvester is None:
            logger.critical(
                "!!! CRITICAL ERROR: Harvester was not initialized by the agent. Halting. !!!"
            )
            return
        harvester = axiom_agent.harvester

        metacognitive_engine = MetacognitiveEngine(
            agent=axiom_agent,
            gemini_api_key=GEMINI_API_KEY,
        )

        scheduler = BackgroundScheduler(daemon=True)
        manager = CycleManager(scheduler, harvester, metacognitive_engine)

        _generate_or_set_initial_goal(axiom_agent, initial_goal)

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

    parser = argparse.ArgumentParser(
        description="Run the Axiom Agent's autonomous training cycles."
    )

    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="A specific, high-level learning goal to start with.",
    )

    parser.add_argument(
        "brain_file",
        type=Path,
        nargs="?",
        default=DEFAULT_BRAIN_FILE,
        help="Path to the agent's brain file.",
    )
    parser.add_argument(
        "state_file",
        type=Path,
        nargs="?",
        default=DEFAULT_STATE_FILE,
        help="Path to the agent's state file.",
    )

    args = parser.parse_args()

    start_autonomous_training(args.brain_file, args.state_file, initial_goal=args.goal)


if __name__ == "__main__":
    main()
