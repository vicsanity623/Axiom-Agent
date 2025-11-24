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
3.  The string MUST contain exactly 6 stages, each formatted as `[Stage X: Title]: Description...`.
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

    max_curriculum_attempts = 3
    for curriculum_attempt in range(max_curriculum_attempts):
        logger.info(
            f"[Curriculum Generation]: Attempt {curriculum_attempt + 1}/{max_curriculum_attempts}"
        )
        creative_seeds = [
            # === ENGLISH LANGUAGE FUNDAMENTALS ===
            # Tier 1: Basic English Building Blocks
            "Generate a curriculum about the fundamental units of English language. Start with letters, sounds, and word formation. Example topics: alphabet, phoneme, vowel, consonant, syllable, word, letter, sound, pronunciation, spelling, morpheme.",
            "Generate a curriculum about English word types and basic sentence structure. Start with parts of speech. Example topics: noun, verb, adjective, adverb, pronoun, preposition, conjunction, sentence, subject, predicate, article.",
            # Tier 2: English Grammar and Semantics
            "Generate a curriculum about how meaning is constructed in English. Start with word meanings and relationships. Example topics: definition, synonym, antonym, homonym, context, meaning, semantics, metaphor, literal, figurative, connotation, denotation.",
            "Generate a curriculum about English sentence construction and grammar rules. Start with tenses and sentence types. Example topics: tense, past, present, future, punctuation, comma, period, question, exclamation, declarative, interrogative.",
            # Tier 3: Advanced English Communication
            "Generate a curriculum about complex English expression and writing. Start with paragraphs and composition. Example topics: paragraph, essay, argument, thesis, evidence, transition, coherence, style, tone, voice, rhetoric, persuasion.",
            # === PYTHON PROGRAMMING FUNDAMENTALS ===
            # Tier 4: Python Primitives and Syntax
            "Generate a curriculum about the atomic building blocks of Python programming. Start with data types and basic syntax. Example topics: python, interpreter, variable, string, integer, float, boolean, type, assignment, literal, expression, statement.",
            "Generate a curriculum about Python operators and expressions. Start with arithmetic and comparison. Example topics: operator, addition, subtraction, multiplication, division, comparison, equals, less than, greater than, boolean, logical, and, or, not.",
            # Tier 5: Python Control Flow
            "Generate a curriculum about controlling program execution in Python. Start with conditional logic and loops. Example topics: if, else, elif, condition, indentation, block, loop, for, while, range, iteration, break, continue, nested.",
            # Tier 6: Python Data Structures
            "Generate a curriculum about organizing and storing data in Python. Start with collections. Example topics: list, tuple, dictionary, set, index, slice, key, value, append, remove, mutable, immutable, sequence, collection.",
            "Generate a curriculum about Python functions and code organization. Start with function definition. Example topics: function, def, return, parameter, argument, scope, local, global, module, import, package, reusability.",
            # Tier 7: Python Object-Oriented Programming
            "Generate a curriculum about Object-Oriented Programming in Python. Start with classes and objects. Example topics: class, object, instance, method, attribute, self, init, constructor, encapsulation, inheritance, polymorphism.",
            # === COMPARATIVE UNDERSTANDING: LANGUAGE vs CODE ===
            # Tier 8: Distinguishing Natural Language from Programming Language
            "Generate a curriculum about understanding the fundamental differences between English and Python. Start with ambiguity versus precision. Example topics: ambiguity, precision, syntax rules, grammar, interpretation, execution, natural language, formal language, semantics, parsing, compiler.",
            "Generate a curriculum about how English and Python each represent and communicate meaning. Start with human communication versus machine instructions. Example topics: intent, instruction, command, interpretation, literal meaning, context-free, context-dependent, abstraction, translation, communication.",
            # Tier 9: Error Handling in Both Domains
            "Generate a curriculum about mistakes and corrections in English versus Python. Start with grammar errors versus syntax errors. Example topics: grammar error, syntax error, typo, bug, exception, debugging, correction, validation, linting, proofreading, traceback.",
            # Tier 10: Structure and Composition
            "Generate a curriculum about how complex structures are built in English and Python. Start with composition principles. Example topics: composition, hierarchy, nesting, paragraph structure, code structure, modularity, organization, refactoring, clarity, readability.",
            # === ADVANCED CONCEPTS ===
            # Tier 11: Advanced Python Features
            "Generate a curriculum about advanced Python programming techniques. Start with iterators and comprehensions. Example topics: iterator, generator, yield, comprehension, lambda, decorator, context manager, with statement, functional programming.",
            # Tier 12: Meta-Understanding of Languages
            "Generate a curriculum about understanding language systems themselves. Start with how languages evolve and are designed. Example topics: language design, evolution, standard, convention, idiom, pattern, best practice, style guide, pythonic, prescriptive, descriptive.",
        ]
        selected_seed = random.choice(creative_seeds)
        logger.info(
            "[purple][Curriculum Generation]: Using creative seed: '%s'[/purple]",
            selected_seed,
        )

        # Get all nodes, but then create a smaller, more strategic summary.
        all_nodes = agent.graph.get_all_nodes()
        if all_nodes:
            # Sort nodes by connectivity (degree), access count, and recency.
            # Well-connected nodes are "hub" concepts that should be known.
            sorted_nodes = sorted(
                all_nodes,
                key=lambda n: (
                    len(
                        agent.graph.get_edges_from_node(n.id)
                    ),  # Connection count (degree)
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
                f"   - Providing LLM with a strategic summary of {len(strategic_summary_nodes)} most relevant concepts (sorted by connectivity)."
            )
        else:
            known_concepts_str = "None"

        prompt = CURRICULUM_GENERATION_PROMPT.format(
            creative_seed=selected_seed, known_concepts=known_concepts_str
        )

        # Retry logic with exponential backoff
        new_curriculum_goal = None
        for retry in range(3):
            try:
                new_curriculum_goal = agent.interpreter.synthesize(
                    structured_facts=prompt, mode="creative_writing"
                )
                if new_curriculum_goal and "[Stage 1:" in new_curriculum_goal:
                    break
                logger.warning(
                    f"[Curriculum Generation]: LLM synthesis attempt {retry + 1}/3 returned invalid format"
                )
            except Exception as e:
                logger.error(
                    f"[Curriculum Generation]: LLM synthesis attempt {retry + 1}/3 failed: {e}"
                )
            if retry < 2:  # Don't sleep on the last attempt
                time.sleep(2**retry)  # Exponential backoff: 1s, 2s

        if not new_curriculum_goal or "[Stage 1:" not in new_curriculum_goal:
            logger.warning(
                f"[Curriculum Generation]: Attempt {curriculum_attempt + 1} failed to generate valid curriculum."
            )
            if curriculum_attempt < max_curriculum_attempts - 1:
                continue
            logger.error(
                "--- [AUTONOMOUS TRAINER]: All curriculum generation attempts failed. Using fallback goal. ---"
            )
            agent.goal_manager.add_goal(
                "Learn fundamental programming concepts.[Stage 1: Basic Syntax]: Understand variables, data types, and basic operators."
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
                logger.info("[Curriculum Generation]: Retrying with different seed...")
                agent.goal_manager.goals.clear()
                continue

            active_goal["sub_goals"] = filtered_sub_goals
            agent.learning_goals[:] = filtered_sub_goals
            break
    else:
        # All attempts exhausted without success
        logger.error(
            "--- [AUTONOMOUS TRAINER]: Maximum curriculum generation attempts reached. Using fallback goal. ---"
        )
        agent.goal_manager.add_goal(
            "Learn fundamental programming concepts.[Stage 1: Basic Syntax]: Understand variables, data types, and basic operators."
        )


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
