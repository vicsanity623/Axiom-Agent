from __future__ import annotations

import argparse
import logging
import random
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Final

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

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

logger = logging.getLogger(__name__)


GRAPH_SCRIPT = PROJECT_ROOT / "src" / "axiom" / "scripts" / "graph_deferred_trends.py"
VENV_ACTIVATE = PROJECT_ROOT / ".venv" / "bin" / "activate"


def _launch_graph_monitor_terminal() -> None:
    """
    Opens a new macOS Terminal window, activates the venv, and runs
    graph_deferred_trends.py live alongside axiom-train.
    """

    project_root = str(PROJECT_ROOT)
    graph_script = str(GRAPH_SCRIPT)
    venv_activate = str(VENV_ACTIVATE)

    # QUOTE EVERYTHING to handle spaces in paths
    command = f'''
        tell application "Terminal"
            activate
            do script "cd \\"{project_root}\\" && source \\"{venv_activate}\\" && python3 \\"{graph_script}\\""
        end tell
    '''

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
3.  The string MUST contain exactly 5 stages, each formatted as `[Stage X: Title]: Description...`.
4.  The stages MUST follow a logical progression from concrete fundamentals to abstract reasoning, starting at a "toddler" level.
5.  **Crucially, you MUST introduce variation and avoid redundancy.** The AI has already learned the concepts listed in the "Known Concepts" section. Your generated curriculum should focus on related, but distinct, topics to broaden the AI's knowledge.

**KNOWN CONCEPTS TO AVOID:**
{known_concepts}

**EXAMPLE OUTPUT (Do NOT copy this example; use it only for structure):**
"Embark on a mission to understand the world through the scientific method, evolving from a simple observer to a systematic investigator.[Stage 1: Empirical Observation]: Learn to identify and describe objective phenomena, distinguishing between objects, properties, and events. [Stage 2: Pattern Recognition]: Analyze collections of observations to identify correlations, sequences, and recurring patterns in the data. [Stage 3: Hypothesis Formulation]: Based on observed patterns, learn to generate simple, testable explanations or 'if-then' hypotheses about cause and effect."
"""


def _generate_or_set_initial_goal(
    agent: CognitiveAgent, custom_goal: str | None
) -> None:
    """
    Sets a custom initial goal or generates a new, state-aware, and potentially
    self-renewing curriculum.
    """
    active_goal = agent.goal_manager.get_active_goal()
    if active_goal:
        logger.info(
            "--- [AUTONOMOUS TRAINER]: Found existing active goals. Resuming previous learning plan. ---"
        )
        # Ensure learning_goals is populated from the active goal's sub-goals
        if active_goal.get("sub_goals"):
            agent.learning_goals = list(active_goal["sub_goals"])
        return

    # --- FIX: Check for pending goals to resume before generating new ones ---
    if agent.goal_manager.activate_next_pending_goal():
        logger.info(
            "--- [AUTONOMOUS TRAINER]: Found pending goals. Resuming saved learning plan. ---"
        )
        # Ensure learning_goals is populated from the newly activated goal
        active_goal = agent.goal_manager.get_active_goal()
        if active_goal and active_goal.get("sub_goals"):
            agent.learning_goals = list(active_goal["sub_goals"])
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
            # Tier 1: The Primitives of Code
            "Generate a curriculum about the atomic building blocks of Python programming. Start with data types and basic syntax. Example topics: python, code, interpreter, variable, string, integer, float, boolean, syntax, comment, assign.",
            "Generate a curriculum about basic operators and logic in programming. Start with math and comparisons. Example topics: add, subtract, multiply, divide, equals, not equals, greater than, true, false, logic, operator.",
            # Tier 2: Control Flow & Logic
            "Generate a curriculum about controlling the flow of execution in software. Start with decisions and loops. Example topics: if statement, else, elif, condition, loop, iteration, for loop, while loop, break, continue, block, indentation.",
            # Tier 3: Data Structures & Organization
            "Generate a curriculum about organizing data in memory using Python structures. Start with lists and mappings. Example topics: list, dictionary, tuple, set, index, key, value, append, pop, element, sequence, mutable, immutable.",
            "Generate a curriculum about reusable code and modularity. Start with functions and scope. Example topics: function, def, return, call, argument, parameter, scope, global, local, module, import, library.",
            # Tier 4: Object-Oriented Programming (OOP)
            "Generate a curriculum about Object-Oriented Programming (OOP) concepts. Start with the blueprint of objects. Example topics: class, object, instance, instantiate, attribute, method, self, init, constructor, property.",
            "Generate a curriculum about advanced OOP relationships. Start with how classes interact. Example topics: inheritance, parent class, child class, polymorphism, override, encapsulation, abstraction, interface, composition.",
            # Tier 5: Code Quality, Error Handling, & Review
            "Generate a curriculum about handling errors and preventing crashes. Start with exceptions. Example topics: error, bug, crash, exception, raise, try, except, finally, traceback, debugging.",
            "Generate a curriculum about writing clean, maintainable, and professional code. Start with style and readability. Example topics: refactoring, readability, linting, pep8, docstring, annotation, type hint, clean code, review, testing.",
            # Tier 6: Advanced Python Concepts
            "Generate a curriculum about advanced Python features for efficient coding. Start with iterators and decorators. Example topics: iterator, generator, yield, decorator, wrapper, context manager, with statement, lambda, list comprehension.",
        ]
        selected_seed = random.choice(creative_seeds)
        logger.info(
            "[purple][Curriculum Generation]: Using creative seed: '%s'[/purple]",
            selected_seed,
        )

        # --- FIX: Replace the "Brain Dump" with a "Strategic Summary" ---
        # Get all nodes, but then create a smaller, more strategic summary.
        all_nodes = agent.graph.get_all_nodes()
        if all_nodes:
            # Sort nodes by a combination of access count and how recent they are.
            # This gives a good proxy for "importance" or "relevance".
            sorted_nodes = sorted(
                all_nodes,
                key=lambda n: (
                    n.properties.get("access_count", 0),
                    n.properties.get("last_accessed", ""),
                ),
                reverse=True,
            )
            # Take a sample of the top 30 most relevant concepts.
            # Reduced to 30 to ensure prompt fits within batch size.
            strategic_summary_nodes = sorted_nodes[:30]
            known_concepts_str = ", ".join(
                sorted(n.name for n in strategic_summary_nodes)
            )
            logger.info(
                f"   - Providing LLM with a strategic summary of {len(strategic_summary_nodes)} most relevant concepts."
            )
        else:
            known_concepts_str = "None"

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

                # This logic for generating a new seed is complex, let's simplify for now
                # creative_seeds = [agent.interpreter.generate_new_creative_seed()]
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
    launch_ui: bool = False,
    stop_event: threading.Event | None = None,
    agent_callback: Callable[[CognitiveAgent], None] | None = None,
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

    if launch_ui:
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

        # Expose agent to caller (e.g. studio server)
        if agent_callback:
            agent_callback(axiom_agent)

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
            stop_event=stop_event,
        )

        _generate_or_set_initial_goal(axiom_agent, initial_goal)

        manager.start(duration_hours=duration_hours)
        scheduler.start()

        while scheduler.running:
            if stop_event and stop_event.is_set():
                logger.info(
                    "[border]\n--- [AUTONOMOUS TRAINER]: External stop signal received. Shutting down... ---[/border]",
                )
                break
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
        "--ui",
        action="store_true",
        help="Launch the legacy external graph monitor UI.",
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
        launch_ui=args.ui,
    )


if __name__ == "__main__":
    main()
