from __future__ import annotations

# autonomous_trainer.py
import threading
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


def start_autonomous_training() -> None:
    """Initialize the agent and run its autonomous learning cycles indefinitely.

    This script runs the Axiom Agent in a "headless" mode, with no user
    interaction. It loads the agent in its learning-enabled state,
    schedules the recurring Study and Discovery cycles, and then enters an
    infinite loop to keep the process alive.

    This is the primary method for enabling the agent's 24/7,
    unattended self-improvement.
    """
    print("--- [AUTONOMOUS TRAINER]: Starting Axiom Agent Initialization... ---")

    axiom_agent: CognitiveAgent | None = None
    agent_interaction_lock = threading.Lock()

    try:
        brain_path = Path("brain")
        brain_file = brain_path / "my_agent_brain.json"
        state_file = brain_path / "my_agent_state.json"

        axiom_agent = CognitiveAgent(
            brain_file=brain_file,
            state_file=state_file,
            inference_mode=False,
        )

        harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)
        scheduler = BackgroundScheduler(daemon=True)

        scheduler.add_job(
            harvester.study_cycle,
            "interval",
            minutes=6,
        )
        print("--- [SCHEDULER]: Study Cycle is scheduled to run every 6 minutes. ---")

        scheduler.add_job(
            harvester.discover_cycle,
            "interval",
            minutes=35,
        )
        print(
            "--- [SCHEDULER]: Discovery Cycle is scheduled to run every 35 minutes. ---",
        )

        scheduler.start()

        print("\n--- [AUTONOMOUS TRAINER]: Agent is running in headless mode. ---")
        print("--- Knowledge Harvester is active. Press CTRL+C to stop. ---")

        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        print("\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Exiting. ---")
    except Exception as exc:
        print(f"!!! [AUTONOMOUS TRAINER]: CRITICAL ERROR: {exc} !!!")
        import traceback

        traceback.print_exc()
    finally:
        print("--- [AUTONOMOUS TRAINER]: Process terminated. ---")


if __name__ == "__main__":
    start_autonomous_training()
