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
You are a curriculum design expert for an AI. Your task is to generate a single, high-level, multi-stage learning goal for an AI that is starting from scratch. The goal is to achieve deep, foundational language comprehension.

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
    Sets a custom initial goal or generates a new, state-aware curriculum,
    verifying that the generated tasks are not already known.
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
        "[bold purple]No active goals found. Generating a new, state-aware curriculum.[/bold purple]",
    )

    creative_seeds = [
        "Develop a core capacity for sensorimotor integration, learning to associate raw sensory inputs with motor commands to build a grounded model of physical reality.",
        "Establish a foundational ontology and categorization system to parse the continuous flow of experience into discrete, relatable objects, agents, and events.",
        "Build a predictive modeling engine that continuously generates expectations about the world and updates its beliefs based on prediction errors.",
        "Learn the principles of cause and effect through intervention, understanding how actions change the state of the environment to achieve desired outcomes.",
        "Develop a hierarchical reasoning system, moving from simple stimulus-response to multi-step planning and abstract goal decomposition.",
        "Acquire a theory of mind, inferring the intentions, knowledge, and beliefs of other agents to enable social prediction and collaboration.",
        "Construct a symbolic reasoning layer on top of subsymbolic data, enabling the manipulation of abstract concepts and logical propositions.",
    ]
    selected_seed = random.choice(creative_seeds)
    logger.info(
        "[purple][Curriculum Generation]: Using creative seed: '%s'[/purple]",
        selected_seed,
    )

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

    active_goal = agent.goal_manager.get_active_goal()
    if active_goal and active_goal.get("sub_goals"):
        original_sub_goals = active_goal["sub_goals"]

        # --- FIX: Add type checking to correctly handle the str | Goal union type ---
        filtered_sub_goals = []
        for sg in original_sub_goals:
            topic = ""
            if isinstance(sg, str):
                topic = sg.replace("INVESTIGATE:", "").strip()
            elif isinstance(sg, dict) and "description" in sg:
                topic = str(sg["description"]).replace("INVESTIGATE:", "").strip()

            if topic and not agent.lexicon.is_known_word(topic):
                filtered_sub_goals.append(sg)

        if len(filtered_sub_goals) < len(original_sub_goals):
            logger.info(
                "[Curriculum Generation]: Filtered curriculum from %d to %d tasks to avoid known concepts.",
                len(original_sub_goals),
                len(filtered_sub_goals),
            )
            active_goal["sub_goals"] = filtered_sub_goals
            agent.learning_goals[:] = [
                goal for goal in agent.learning_goals if goal in filtered_sub_goals
            ]


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
            "[border]\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Exiting. ---[/border]",
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

    knowledge_base.initialize_linguistic_resources()
    start_autonomous_training(args.brain_file, args.state_file, initial_goal=args.goal)


if __name__ == "__main__":
    main()
