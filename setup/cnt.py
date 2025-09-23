from __future__ import annotations

# app.py (Command-Line Manual Trainer)
from axiom.cognitive_agent import CognitiveAgent


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

    try:
        axiom_agent = CognitiveAgent(inference_mode=False)
        print("--- [TRAINER]: Agent initialized. You can now begin training. ---")
        print("--- [TRAINER]: Type 'quit' or 'exit' to save and end the session. ---")
    except Exception as e:
        print(f"!!! [TRAINER]: CRITICAL ERROR during initialization: {e} !!!")
        import traceback

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
        except Exception as e:
            print(f"!!! [TRAINER]: An error occurred during the chat loop: {e} !!!")


if __name__ == "__main__":
    run_training_session()
