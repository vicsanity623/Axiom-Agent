""" Test the metacognitive Engine to ensure API keys are active"""

from __future__ import annotations

import logging
import os

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE, GEMINI_API_KEY
from ..logging_config import setup_logging
from ..metacognitive_engine import MetacognitiveEngine

logger = logging.getLogger(__name__)


def run_single_metacognitive_cycle():
    """
    A dedicated test harness to initialize the MetacognitiveEngine and run
    a single introspection cycle for debugging and testing.
    """
    logger.info("--- [METACOG TEST HARNESS]: Starting single-cycle test... ---")

    # 1. Check for the required API key.
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.critical(
            "!!! CRITICAL ERROR: GEMINI_API_KEY environment variable is not set. Halting. !!!"
        )
        return

    # 2. Perform minimal initialization of the agent and the engine.
    try:
        logger.info("--- [METACOG TEST HARNESS]: Initializing Cognitive Agent... ---")
        axiom_agent = CognitiveAgent(
            brain_file=DEFAULT_BRAIN_FILE,
            state_file=DEFAULT_STATE_FILE,
            inference_mode=True,  # Run in inference mode to prevent brain saves, etc.
        )

        logger.info("--- [METACOG TEST HARNESS]: Initializing Metacognitive Engine... ---")
        metacognitive_engine = MetacognitiveEngine(
            agent=axiom_agent,
            gemini_api_key=gemini_api_key,
        )

        # 3. Directly call the target function.
        metacognitive_engine.run_introspection_cycle()

    except Exception as e:
        logger.critical(
            "--- [METACOG TEST HARNESS]: An unexpected error occurred during setup or execution: %s ---",
            e,
            exc_info=True,
        )
    finally:
        logger.info("--- [METACOG TEST HARNESS]: Single-cycle test finished. ---")


def main():
    """Entry point for the axiom-metatest command."""
    setup_logging()
    run_single_metacognitive_cycle()


if __name__ == "__main__":
    main()