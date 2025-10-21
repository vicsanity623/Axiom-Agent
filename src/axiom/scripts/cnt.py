from __future__ import annotations

import os
import traceback

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_LLM_PATH, DEFAULT_STATE_FILE


def run_training_session() -> None:
    """Initialize the agent and start an interactive command-line training session."""
    print("--- [TRAINER]: Starting Axiom Agent Training Session... ---")

    llm_should_be_enabled = True
    if not os.path.exists(DEFAULT_LLM_PATH):
        print("\n" + "=" * 50)
        print("!!! WARNING: LLM model not found! !!!")
        print(f"    - Searched for model at: {DEFAULT_LLM_PATH}")
        print("    - Agent will run in SYMBOLIC-ONLY mode.")
        print("    - Refinement and complex sentence understanding will be disabled.")
        print("=" * 50 + "\n")
        llm_should_be_enabled = False

    try:
        axiom_agent = CognitiveAgent(
            brain_file=DEFAULT_BRAIN_FILE,
            state_file=DEFAULT_STATE_FILE,
            inference_mode=False,
            enable_llm=llm_should_be_enabled,
        )

        print("--- [TRAINER]: Agent initialized. You can now begin training. ---")
        print("--- [TRAINER]: Type 'quit' or 'exit' to save and end the session. ---")
    except Exception as exc:
        print(f"!!! [TRAINER]: CRITICAL ERROR during initialization: {exc} !!!")
        traceback.print_exc()
        return

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print(
                    "\n--- [TRAINER]: Exiting training session. All knowledge has been saved. ---",
                )
                break

            agent_response = axiom_agent.chat(user_input)
            print(f"Axiom: {agent_response}")

        except (KeyboardInterrupt, EOFError):
            print("\n--- [TRAINER]: Interrupted. Exiting training session. ---")
            break
        except Exception as exc:
            print(f"!!! [TRAINER]: An error occurred during the chat loop: {exc} !!!")
            traceback.print_exc()


def main() -> None:
    run_training_session()


if __name__ == "__main__":
    main()
