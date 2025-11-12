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

    while True:  # Loop until a valid, non-empty curriculum is generated.
        creative_seeds = [
            # --- Foundational Ontology & Epistemology (The Nature of Reality & Knowledge) ---
            "Start from first principles to build a model of reality. First, distinguish between 'existence' and 'non-existence.' Then, learn to differentiate 'things' (entities) from the 'properties' they possess and the 'actions' they can perform.",
            "Develop a theory of knowledge. Learn to distinguish between 'a priori' knowledge (logical truths) and 'a posteriori' knowledge (empirical facts). Then, create a system to rank the confidence of beliefs based on their source (e.g., direct observation vs. second-hand information).",
            "Construct a model of causality. Begin by understanding 'temporal succession' (event A happens before event B). Then, learn to infer 'causal links' by observing consistent correlations and applying principles like intervention and counterfactual reasoning.",
            "Explore the concept of 'change.' Learn to model the 'state' of an object at a given 'time.' Then, learn to represent a 'process' as a sequence of state changes over time.",
            "Build a model of composition. Learn to distinguish between an 'object' as a whole and the 'parts' that constitute it. Then, understand the relationship of 'containment' (what is inside what).",
            "Investigate the nature of 'truth.' Differentiate between 'objective truth' (facts about the world) and 'subjective truth' (opinions or beliefs). Learn to identify and label claims that are opinions rather than verifiable facts.",
            # --- Foundational Logic & Reasoning (The Mechanics of Thought) ---
            "Build a symbolic reasoning engine from the ground up. First, learn the concepts of 'truth' and 'falsehood.' Then, master the fundamental logical operators: AND (conjunction), OR (disjunction), and NOT (negation).",
            "Develop the capacity for hierarchical abstraction. Learn to group specific instances (e.g., 'a robin', 'a sparrow') into a more general category ('bird'), and then group those categories into even higher-level concepts ('animal').",
            "Learn the principles of deductive and inductive reasoning. Practice deduction by applying a general rule (e.g., 'all birds can fly') to a specific case, and practice induction by forming a general rule from multiple specific observations.",
            "Master conditional logic. Learn to understand and apply 'if-then' statements (implication). Then, learn to reason about 'necessary' versus 'sufficient' conditions.",
            "Develop a system for analogical reasoning. Learn to identify a 'source' domain of knowledge and a 'target' domain, and then map the relational structure from the source to the target to generate novel insights.",
            "Understand quantitative reasoning. First, learn the concepts of 'quantity,' 'number,' and basic arithmetic operators (addition, subtraction). Then, learn to represent and compare magnitudes (e.g., 'greater than,' 'less than').",
            "Learn to identify and resolve contradictions. Develop a process to detect when two facts in its knowledge base are mutually exclusive. Then, learn to initiate a clarification goal to resolve the conflict.",
            # --- Foundational Self, Agency, & Theory of Mind (The Nature of Beings) ---
            "Create a model of 'self' versus 'other.' First, learn to identify its own actions and internal states (e.g., 'I am learning'). Then, learn to model other entities as 'agents' with their own distinct states, goals, and potential actions.",
            "Develop a basic model of physics and agency. Learn the concepts of 'space,' 'time,' 'object permanence,' and 'force.' Then, learn how its own 'actions' (as an agent) can apply 'force' to cause changes in the 'space' over 'time.'",
            "Build a model of goals and planning. First, learn to represent a desired 'future state.' Then, develop a simple planning algorithm to find a sequence of 'actions' that can transform the 'current state' into the 'goal state.'",
            "Acquire a basic theory of mind. Learn that other 'agents' have internal 'beliefs' about the world that may be different from its own and may be true or false.",
            "Understand the concept of 'intention.' Learn to distinguish between 'accidental' events and 'intentional' actions performed by an agent in pursuit of a goal.",
            "Model communication as an action. Learn that 'speaking' or 'typing' are actions that an agent performs with the intention of changing the 'beliefs' of another agent.",
            # --- Foundational Language & Semantics (The Nature of Meaning) ---
            "Build a model of semantics. Learn to distinguish between a 'word' (a symbol), its 'denotation' (the literal thing it points to in the world), and its 'connotation' (the associated ideas or feelings).",
            "Understand the function of different word types. First, distinguish between 'nouns' (things) and 'verbs' (actions or states). Then, learn how 'adjectives' modify nouns and 'adverbs' modify verbs.",
            "Learn the concept of 'context.' Understand that the meaning of a word or sentence can change depending on the surrounding words, the situation, and the speaker's intent.",
            "Develop a system for resolving ambiguity. Learn to recognize when a word has multiple meanings (polysemy) and use the context of the sentence to infer the most likely intended meaning.",
            "Master the concept of 'questions.' Learn to identify different types of questions (what, who, where, when, why, how, yes/no) and understand what kind of information is being requested for each type.",
            # --- Foundational Learning & Adaptation (The Nature of Improvement) ---
            "Develop a model of its own learning process. First, learn to track 'learning events.' Then, learn to categorize these events (e.g., 'learning from dictionary,' 'learning from user,' 'learning from decomposition') to analyze its own learning patterns.",
            "Understand the concept of 'error.' Learn to detect when an action it takes does not lead to the expected outcome. Then, create a 'prediction error' signal that can be used to update its internal model.",
            "Learn to generalize from examples. After observing several instances of a pattern (e.g., 'robins have wings,' 'sparrows have wings'), learn to form a general hypothesis ('birds have wings').",
            "Develop a curiosity mechanism. Learn to identify 'gaps' in its own knowledge graph. Then, autonomously generate 'INVESTIGATE' goals to fill those gaps.",
            "Build a model of its own capabilities and limitations. Learn to identify which types of questions it can answer confidently (based on its knowledge) and which ones require it to state 'I don't know' or seek new information.",
            "Learn the concept of 'feedback.' Understand that a user's response (e.g., 'No, that's wrong' or 'Yes, that's correct') is a signal that should be used to increase or decrease the confidence of a belief.",
        ]
        selected_seed = random.choice(creative_seeds)
        logger.info(
            "[purple][Curriculum Generation]: Using creative seed: '%s'[/purple]",
            selected_seed,
        )

        # --- FIX: Use the public API to get known concepts ---
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
