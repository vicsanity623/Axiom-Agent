from __future__ import annotations

import argparse
import logging
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Final

from apscheduler.schedulers.background import BackgroundScheduler

from ..cognitive_agent import CognitiveAgent
from ..config import (
    DEFAULT_BRAIN_FILE,
    DEFAULT_STATE_FILE,
    GEMINI_API_KEY,
    PROJECT_ROOT,
)
from ..logging_config import console, setup_logging
from ..metacognitive_engine import MetacognitiveEngine
from .cycle_manager import CycleManager

logger = logging.getLogger(__name__)


GRAPH_SCRIPT = PROJECT_ROOT / "src" / "axiom" / "scripts" / "graph_deferred_trends.py"
VENV_ACTIVATE = PROJECT_ROOT / ".venv" / "bin" / "activate"


def _launch_graph_monitor_terminal() -> None:
    """
    Opens a new macOS Terminal window, activates the venv, and runs
    graph_deferred_trends.py live alongside axiom-train.
    """

    command = f"""
        tell application "Terminal"
            activate
            do script "cd {PROJECT_ROOT} && source {VENV_ACTIVATE} && python3 {GRAPH_SCRIPT}"
        end tell
    """

    try:
        subprocess.Popen(["osascript", "-e", command])
        logger.info(
            "[AUTONOMOUS TRAINER]: Live graph monitor launched in new Terminal."
        )
    except Exception as e:
        logger.error(f"Failed to launch graph monitor terminal: {e}")


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
            # -------------------------------------- AXIOM "CONCEPTUAL" CURRICULUM SEEDS --------------------------------------
            # --- Foundational Physics & Causality ---
            "Model the physics of a thrown ball. A person can throw a ball. A ball is a round object. Gravity is a force that pulls the ball down. The ball will fall to the ground.",
            "Explore the concept of temperature. Fire is hot. Ice is cold. Water can be hot or cold. Hot things can burn you. Cold things can freeze water.",
            "Understand the states of matter. Water is a liquid. Ice is a solid. Steam is a gas. Heat can turn ice into water. Heat can turn water into steam.",
            # --- Simple Biology & Taxonomies ---
            "Build a model of a simple ecosystem. A cat is an animal. A dog is an animal. A bird is an animal. A cat eats a mouse. A bird eats a worm. Animals need to eat food to live.",
            "Learn about the parts of a plant. A tree is a plant. A tree has roots, a trunk, and leaves. The roots are in the ground. The leaves grow on branches.",
            "Discover the life cycle of a butterfly. A caterpillar is an insect. A caterpillar turns into a chrysalis. A chrysalis becomes a butterfly. A butterfly has wings.",
            # --- Social Structures & Roles ---
            "Model a simple family unit. A family can have a mother and a father. A mother is a parent. A father is a parent. Parents take care of a child.",
            "Understand community helpers. A doctor is a person who helps sick people. A firefighter is a person who puts out fires. A teacher is a person who teaches students.",
            "Explore the concept of a city. A city is a place where many people live. A city has buildings, roads, and parks. Cars drive on roads.",
            # --- Abstract Concepts & Properties ---
            "Learn about colors and their associations. The sun is yellow. The sky is blue. An apple is red. Red can mean stop. Green can mean go.",
            "Understand the concept of size and scale. An elephant is big. A mouse is small. A mountain is very big. A building is bigger than a house.",
            "Explore the concept of time. The day has a morning, an afternoon, and a night. The sun is in the sky during the day. The moon is in the sky at night.",
            # --- Tools & Purpose ---
            "Model the purpose of simple tools. A hammer is a tool used to hit nails. A pencil is a tool used to write. A spoon is a tool used to eat soup.",
            "Understand simple machines. A wheel is a simple machine that can roll. A lever is a simple machine that can lift heavy things.",
            "Explore the concept of a container. A box is a container. A cup is a container. A bottle is a container. A container is used to hold things.",
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
    brain_file: Path,
    state_file: Path,
    initial_goal: str | None = None,
    duration_hours: float | None = None,
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

    # Launch the trends graph in a separate macOS Terminal window
    _launch_graph_monitor_terminal()

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
                "!!! CRITICAL ERROR: Harvester not initialized. Halting. !!!"
            )
            return
        harvester = axiom_agent.harvester

        metacognitive_engine = MetacognitiveEngine(
            agent=axiom_agent, gemini_api_key=GEMINI_API_KEY
        )

        scheduler = BackgroundScheduler(daemon=True)
        manager = CycleManager(
            scheduler=scheduler,
            harvester=harvester,
            metacognitive_engine=metacognitive_engine,
            console=console,
        )

        _generate_or_set_initial_goal(axiom_agent, initial_goal)

        # FIX: Pass the duration to the manager's start method
        manager.start(duration_hours=duration_hours)
        scheduler.start()

        # The main loop now just waits for the scheduler to be shut down
        while scheduler.running:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        logger.info(
            "[border]\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Saving final state... ---[/border]",
        )
    except Exception as exc:
        logger.critical(
            "!!! [AUTONOMOUS TRAINER]: CRITICAL ERROR: %s !!!", exc, exc_info=True
        )
        traceback.print_exc()
    finally:
        if axiom_agent:
            logger.info("--- [AUTONOMOUS TRAINER]: Attempting to save state... ---")
            axiom_agent.save_state()
            logger.info("--- [AUTONOMOUS TRAINER]: Final state saved. Exiting. ---")
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
    # --- FIX: Add the new --duration argument ---
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="An optional duration in hours to run the training. e.g., 1.5 for 90 minutes.",
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

    # FIX: Pass the new duration argument to the training function
    start_autonomous_training(
        args.brain_file,
        args.state_file,
        initial_goal=args.goal,
        duration_hours=args.duration,
    )


if __name__ == "__main__":
    main()
