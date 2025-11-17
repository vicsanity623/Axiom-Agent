from __future__ import annotations

import argparse
import logging
import random
import time
import traceback
from pathlib import Path
from typing import Final

from apscheduler.schedulers.background import BackgroundScheduler

from .. import knowledge_base
from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE, GEMINI_API_KEY
from ..logging_config import console, setup_logging
from ..metacognitive_engine import MetacognitiveEngine
from .cycle_manager import CycleManager

logger = logging.getLogger(__name__)

CURRICULUM_GENERATION_PROMPT: Final[str] = """
You are a curriculum design expert for an AI. Your task is to generate a single, high-level, multi-stage learning goal for an AI that is starting from scratch. The goal is to achieve deep, foundational curriculum comprehension.

**INSPIRATIONAL SEED (Use this to guide your curriculum's theme):**
{creative_seed}

**RULES:**
1.  The output MUST be a single, continuous string.
2.  The string MUST start with a high-level mission statement.
3.  The string MUST contain exactly 7 stages, each formatted as `[Stage X: Title]: Description...`.
4.  The stages MUST follow a logical progression from concrete fundamentals to abstract reasoning, starting at a "toddler" level.
5.  **Crucially, you MUST introduce variation and avoid redundancy.** The AI has already learned the concepts listed in the "Known Concepts" section. Your generated curriculum should focus on related, but distinct, topics to broaden the AI's knowledge.

**KNOWN CONCEPTS TO AVOID:**
{known_concepts}

**EXAMPLE OUTPUT (Do NOT copy this example; use it only for structure):**
"Embark on a mission to understand the world through the scientific method, evolving from a simple observer to a systematic investigator.[Stage 1: Empirical Observation]: Learn to identify and describe objective phenomena, distinguishing between objects, properties, and events. [Stage 2: Pattern Recognition]: Analyze collections of observations to identify correlations, sequences, and recurring patterns in the data. [Stage 3: Hypothesis Formulation]: Based on observed patterns, learn to generate simple, testable explanations or 'if-then' hypotheses about cause and effect. [Stage 4: Variable Isolation]: Understand the concept of variables and how to distinguish between independent, dependent, and confounding variables in a system. [Stage 5: Experimental Design]: Learn to construct a logical procedure to test a hypothesis by manipulating an independent variable and measuring its effect on a dependent variable. [Stage 6: Evidence Evaluation]: Analyze the results of an experiment to determine whether the evidence supports, refutes, or is inconclusive regarding the hypothesis. [Stage 7: Theory Building]: Learn to synthesize multiple, validated hypotheses into a broader, more comprehensive explanatory model or theory."
"""


def _generate_or_set_initial_goal(
    agent: CognitiveAgent, custom_goal: str | None
) -> None:
    """
    Sets a custom initial goal or generates a new, state-aware, and potentially
    self-renewing curriculum.
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

    console.rule(
        "[bold purple]No active goals found. Generating a new curriculum.[/bold purple]",
    )

    while True:
        creative_seeds = [
            # -------------------------------------- CHANGE OR CUSTOMIZE WHT AXIOM LEARNS HERE --------------------------------------
            # --- Basic Objects & The World ---
            "Learn about the things in a house. Start with 'chair' and 'table.' Then, learn about 'door' and 'window.' Finally, understand what a 'roof' is.",
            "Explore the kitchen. Learn about a 'cup,' a 'plate,' and a 'spoon.' Then, find out what a 'fork' and a 'knife' are used for.",
            "Learn about things you can wear. Start with 'shirt' and 'pants.' Then, learn about 'shoes' and 'socks.' Finally, understand what a 'hat' is.",
            "Discover things in a classroom. Learn about a 'book,' a 'pencil,' and 'paper.' Then, understand what a 'desk' and a 'blackboard' are.",
            "Learn about toys. Start with 'ball' and 'doll.' Then, learn about 'blocks' and 'crayons.'",
            # --- Colors & Shapes ---
            "Build a model of colors. First, learn the primary colors: 'red,' 'blue,' and 'yellow.' Then, learn the secondary colors: 'green,' 'orange,' and 'purple.'",
            "Learn to identify basic shapes. Start with 'circle' and 'square.' Then, learn about 'triangle' and 'star.'",
            "Explore the colors of nature. Learn that the 'sky' is 'blue' and 'grass' is 'green.' Then, learn that a 'sunflower' is 'yellow.'",
            # --- Animals ---
            "Learn about common pets. Start with 'dog' and 'cat.' Then, learn about 'fish' and 'bird.'",
            "Discover farm animals. Learn about a 'cow,' a 'pig,' and a 'sheep.' Then, find out what a 'chicken' and a 'horse' are.",
            "Explore wild animals. Learn about a 'lion,' a 'tiger,' and a 'bear.' Then, learn about an 'elephant' and a 'monkey.'",
            # --- Simple Actions & Verbs ---
            "Understand basic actions. Learn the meaning of 'walk' and 'run.' Then, learn about 'jump' and 'sleep.'",
            "Learn about how we interact with objects. Start with 'eat' and 'drink.' Then, learn about 'play' and 'read.'",
            "Explore verbs of creation. Learn the meaning of 'draw' and 'build.'",
            # --- People & Family ---
            "Build a model of a family. Learn the concepts of 'mother' and 'father.' Then, learn about 'brother' and 'sister.' Finally, understand 'baby.'",
            "Learn about people in the community. Start with 'teacher' and 'doctor.' Then, learn about 'firefighter' and 'police officer.'",
            # --- Nature & The Outdoors ---
            "Learn about the sky. Start with the 'sun' and the 'moon.' Then, learn about 'stars' and 'clouds.'",
            "Explore the weather. Learn the difference between 'rain' and 'snow.' Then, understand what 'wind' is.",
            "Discover plants. Learn about a 'tree,' a 'flower,' and 'grass.'",
            # --- Food ---
            "Learn about fruits. Start with 'apple' and 'banana.' Then, learn about 'orange' and 'grape.'",
            "Discover vegetables. Learn about 'carrot' and 'pea.' Then, learn about 'broccoli' and 'corn.'",
            "Learn about simple meals. Start with 'bread' and 'cheese.' Then, learn about 'milk' and 'water.'",
            # --- Basic Concepts ---
            "Understand opposites. Learn the difference between 'big' and 'small.' Then, learn about 'hot' and 'cold,' and 'up' and 'down.'",
            "Learn to count. Start with the numbers 'one,' 'two,' and 'three.' Then, learn 'four' and 'five.'",
            "Explore the senses. Learn about 'seeing,' 'hearing,' and 'touching.' Then, learn about 'smelling' and 'tasting.'",
        ]
        selected_seed = random.choice(creative_seeds)
        logger.info(
            "[purple][Curriculum Generation]: Using creative seed: '%s'[/purple]",
            selected_seed,
        )

        known_concepts = agent.graph.get_all_node_names()
        known_concepts_str = (
            ", ".join(sorted(known_concepts)) if known_concepts else "None"
        )

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

        active_goal = agent.goal_manager.get_active_goal()
        if active_goal and active_goal.get("sub_goals"):
            original_sub_goals = active_goal["sub_goals"]

            filtered_sub_goals = [
                sg
                for sg in original_sub_goals
                if not agent.conversation_manager.is_concept_well_known(
                    str(sg).replace("INVESTIGATE:", "").strip()
                )
            ]

            if len(filtered_sub_goals) < len(original_sub_goals):
                logger.info(
                    "[Curriculum Generation]: Filtered curriculum from %d to %d tasks to avoid known concepts.",
                    len(original_sub_goals),
                    len(filtered_sub_goals),
                )

            if not filtered_sub_goals and original_sub_goals:
                logger.warning(
                    "[Curriculum Generation]: All generated topics are already well-known."
                )
                logger.info(
                    "[Curriculum Generation]: Triggering autonomous curiosity to find a new topic..."
                )

                creative_seeds = [agent.interpreter.generate_new_creative_seed()]
                agent.goal_manager.goals.clear()
                continue

            active_goal["sub_goals"] = filtered_sub_goals
            agent.learning_goals[:] = filtered_sub_goals
            break


def start_autonomous_training(
    brain_file: Path, state_file: Path, initial_goal: str | None = None
) -> None:
    """Initialize the agent and run its autonomous learning cycles indefinitely."""

    console.rule(
        "[bold cyan]Starting Axiom Agent Initialization[/bold cyan]", style="border"
    )
    console.rule(
        "[bold green]Agent is running in headless mode.[/bold green]",
        style="border",
    )
    console.rule(
        "[bold green]All learning cycles are active. Press CTRL+C to stop.[/bold green]",
        style="border",
    )

    axiom_agent: CognitiveAgent | None = None
    scheduler: BackgroundScheduler | None = None

    try:
        axiom_agent = CognitiveAgent(
            brain_file=brain_file,
            state_file=state_file,
            inference_mode=False,
        )

        axiom_agent.load_state()

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
        manager = CycleManager(
            scheduler=scheduler,
            harvester=harvester,
            metacognitive_engine=metacognitive_engine,
            console=console,
        )

        _generate_or_set_initial_goal(axiom_agent, initial_goal)

        manager.start()
        scheduler.start()

        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        logger.info(
            "[border]\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Saving final state... ---[/border]",
        )
        if axiom_agent:
            axiom_agent.save_state()
            logger.info("--- [AUTONOMOUS TRAINER]: Final state saved. Exiting. ---")

    except Exception as exc:
        logger.critical(
            "!!! [AUTONOMOUS TRAINER]: CRITICAL ERROR: %s !!!",
            exc,
            exc_info=True,
        )
        traceback.print_exc()
        if axiom_agent:
            logger.info(
                "--- [AUTONOMOUS TRAINER]: Attempting to save state after critical error... ---"
            )
            axiom_agent.save_state()

    finally:
        if scheduler and scheduler.running:
            scheduler.shutdown()
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

    knowledge_base.initialize_linguistic_resources()
    start_autonomous_training(args.brain_file, args.state_file, initial_goal=args.goal)


if __name__ == "__main__":
    main()
