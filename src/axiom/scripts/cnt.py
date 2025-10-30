from __future__ import annotations

import logging
import os

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_LLM_PATH, DEFAULT_STATE_FILE
from ..logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_training_session() -> None:
    """Initialize the agent and start an interactive command-line training session."""
    logger.info("--- [TRAINER]: Starting Axiom Agent Training Session... ---")

    llm_should_be_enabled = True
    if not os.path.exists(DEFAULT_LLM_PATH):
        logger.warning("\n" + "=" * 50)
        logger.warning("!!! LLM model not found! !!!")
        logger.warning("    - Searched for model at: %s", DEFAULT_LLM_PATH)
        logger.warning("    - Agent will run in SYMBOLIC-ONLY mode.")
        logger.warning(
            "    - Refinement and complex sentence understanding will be disabled."
        )
        logger.warning("=" * 50 + "\n")
        llm_should_be_enabled = False

    try:
        axiom_agent = CognitiveAgent(
            brain_file=DEFAULT_BRAIN_FILE,
            state_file=DEFAULT_STATE_FILE,
            inference_mode=False,
            enable_llm=llm_should_be_enabled,
        )

        logger.info("--- [TRAINER]: Agent initialized. You can now begin training. ---")
        logger.info(
            "--- [TRAINER]: Type 'quit' or 'exit' to save and end the session. ---"
        )
    except Exception as exc:
        logger.critical(
            "!!! [TRAINER]: CRITICAL ERROR during initialization: %s !!!",
            exc,
            exc_info=True,
        )
        return

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                logger.info(
                    "\n--- [TRAINER]: Exiting training session. All knowledge has been saved. ---",
                )
                break

            agent_response = axiom_agent.chat(user_input)
            logger.info("Axiom: %s", agent_response)

        except (KeyboardInterrupt, EOFError):
            logger.info("\n--- [TRAINER]: Interrupted. Exiting training session. ---")
            break
        except Exception as exc:
            logger.error(
                "!!! [TRAINER]: An error occurred during the chat loop: %s !!!",
                exc,
                exc_info=True,
            )


def main() -> None:
    """Entry point for the axiom-teach command."""
    setup_logging()
    run_training_session()


if __name__ == "__main__":
    main()
