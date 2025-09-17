# app.py (Command-Line Manual Trainer)

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axiom.cognitive_agent import CognitiveAgent


def run_training_session():
    """
    Initializes the agent and starts a command-line interface for training.
    """
    print("--- [TRAINER]: Starting Axiom Agent Training Session... ---")

    # Initialize the agent. It will load from my_agent_brain.json by default.
    # This is the "training" instance of the agent.
    try:
        axiom_agent = CognitiveAgent(inference_mode=False)
        print("--- [TRAINER]: Agent initialized. You can now begin training. ---")
        print("--- [TRAINER]: Type 'quit' or 'exit' to save and end the session. ---")
    except Exception as e:
        print(f"!!! [TRAINER]: CRITICAL ERROR during initialization: {e} !!!")
        import traceback

        traceback.print_exc()
        return

    # Start the command-line chat loop for training
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print(
                    "\n--- [TRAINER]: Exiting training session. All knowledge has been saved. ---"
                )
                break

            # The agent's chat method automatically saves the brain after learning.
            agent_response = axiom_agent.chat(user_input)
            print(f"Axiom: {agent_response}")

        except (KeyboardInterrupt, EOFError):
            print("\n--- [TRAINER]: Interrupted. Exiting training session. ---")
            break
        except Exception as e:
            print(f"!!! [TRAINER]: An error occurred during the chat loop: {e} !!!")


if __name__ == "__main__":
    run_training_session()
