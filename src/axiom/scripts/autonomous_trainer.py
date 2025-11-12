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
            # --- Kindergarten Level: The Concrete World ---
            "Learn about the world of animals. Start by identifying common pets like 'dogs' and 'cats.' Then, explore farm animals like 'cows' and 'chickens.' Finally, learn about wild animals found in a 'forest' or 'jungle.'",
            "Explore the colors of the rainbow. Learn to identify basic colors like 'red,' 'blue,' and 'yellow.' Then, understand how they combine to make 'green,' 'orange,' and 'purple.'",
            "Understand basic shapes. Start with simple shapes like a 'circle,' 'square,' and 'triangle.' Then, learn to identify these shapes in everyday objects like a 'wheel' or a 'house.'",
            "Learn about the family. Understand the roles of a 'mother,' 'father,' 'brother,' and 'sister.' Explore the concept of a 'home' as a place where a family lives.",
            "Discover the five senses. Learn about 'sight' (eyes), 'hearing' (ears), 'smell' (nose), 'taste' (tongue), and 'touch' (skin) and what they are used for.",
            # --- Elementary School Level: Building Foundational Knowledge ---
            "Build a model of the natural world. Start with the basic elements: 'land,' 'water,' and 'air.' Then, learn about the 'sun,' 'moon,' and 'stars.' Finally, explore weather concepts like 'rain' and 'snow.'",
            "Understand basic biology. Learn what makes something 'living' versus 'non-living.' Explore the life cycle of a 'plant,' from a 'seed' to a 'flower.' Learn about the basic needs of living things: 'food,' 'water,' and 'air.'",
            "Explore the community. Learn about different jobs people have, like a 'doctor,' 'teacher,' and 'firefighter.' Understand the purpose of important places in a town, like a 'school,' 'hospital,' and 'store.'",
            "Learn about basic physics through everyday objects. Understand concepts like 'gravity' (things fall down), 'force' (pushing and pulling), and the states of matter ('solid,' 'liquid,' 'gas').",
            "Build a simple timeline of history. Start with the concept of the 'past,' 'present,' and 'future.' Learn about major historical eras like the 'dinosaurs,' 'ancient civilizations,' and the 'space age.'",
            # --- Middle School Level: Systems and Processes ---
            "Construct a model of the solar system. Learn the names of the planets and their order from the 'sun.' Understand the difference between a 'planet,' a 'moon,' and a 'star.'",
            "Investigate the Earth's ecosystems. Learn about different biomes like the 'ocean,' 'desert,' and 'rainforest.' Understand the concept of a 'food chain' and the roles of 'producers,' 'consumers,' and 'decomposers.'",
            "Explore the fundamentals of government. Learn about the difference between a 'city,' 'state,' and 'country.' Understand basic concepts like 'laws,' 'leaders,' and 'citizens.'",
            "Understand the basics of economics. Learn about 'money,' 'goods,' and 'services.' Explore the concepts of 'buying,' 'selling,' and 'jobs.'",
            "Learn about the human body. Identify major organ systems like the 'skeletal system' (bones), 'circulatory system' (heart and blood), and 'respiratory system' (lungs and breathing).",
            # --- High School Level: Abstract and Complex Systems ---
            "Build a foundational understanding of chemistry. Learn about 'atoms' and 'molecules.' Understand the difference between 'elements' and 'compounds.' Explore simple chemical reactions like 'rusting' or 'burning.'",
            "Investigate the principles of electricity. Learn about 'circuits,' 'batteries,' and 'current.' Understand the difference between 'conductors' and 'insulators.'",
            "Explore the digital world. Learn about the basic components of a 'computer,' such as the 'CPU,' 'memory,' and 'storage.' Understand the concepts of 'software,' 'hardware,' and the 'internet.'",
            "Understand the fundamentals of light and sound. Learn that both are forms of 'energy' that travel in 'waves.' Explore properties like 'color,' 'pitch,' and 'volume.'",
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
                if not agent.is_concept_well_known(
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
