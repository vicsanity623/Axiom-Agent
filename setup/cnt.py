from __future__ import annotations

import os
import traceback
from pathlib import Path

# cnt.py (Command-Line Manual Trainer)
from axiom.cognitive_agent import CognitiveAgent
from axiom.universal_interpreter import DEFAULT_MODEL_PATH

BRAIN_FILE = Path("brain/my_agent_brain.json")
STATE_FILE = Path("brain/my_agent_state.json")


def run_training_session() -> None:
    """Initialize the agent and start an interactive command-line training session.

    This script provides a direct, text-based interface for manually
    interacting with and teaching the Axiom Agent. It loads the agent in
    its learning-enabled state and enters a read-eval-print loop (REPL)
    that allows a user to chat with the agent.

    All new knowledge learned during the session is saved to the brain
    file in real-time.
    """
    print("--- [TRAINER]: Starting Axiom Agent Training Session... ---")

    llm_should_be_enabled = True
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print("\n" + "=" * 50)
        print("!!! WARNING: LLM model not found! !!!")
        print(f"    - Searched for model at: {DEFAULT_MODEL_PATH}")
        print("    - Agent will run in SYMBOLIC-ONLY mode.")
        print("    - Refinement and complex sentence understanding will be disabled.")
        print("=" * 50 + "\n")
        llm_should_be_enabled = False

    try:
        axiom_agent = CognitiveAgent(
            brain_file=BRAIN_FILE,
            state_file=STATE_FILE,
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


if __name__ == "__main__":
    run_training_session()
