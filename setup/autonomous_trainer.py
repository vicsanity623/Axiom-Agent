# autonomous_trainer.py (auto train study and discovery cycle no chat)

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


def start_autonomous_training():
    """
    Initializes the CognitiveAgent and starts the background harvester cycles
    for continuous, unattended learning.
    """
    print("--- [AUTONOMOUS TRAINER]: Starting Axiom Agent Initialization... ---")

    # --- Global Agent Initialization ---
    # In a script, these can be local variables passed around.
    axiom_agent = None
    agent_interaction_lock = threading.Lock()

    try:
        brain_file = "brain/my_agent_brain.json"
        state_file = "brain/my_agent_state.json"

        # We pass inference_mode=False to ensure it can learn and save.
        axiom_agent = CognitiveAgent(
            brain_file=brain_file,
            state_file=state_file,
            inference_mode=False,
        )

        harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)
        scheduler = BackgroundScheduler(daemon=True)

        # --- The Cognitive Scheduler with two independent cycles ---
        # 1. The "Study" cycle runs frequently to deepen existing knowledge.
        scheduler.add_job(harvester.study_existing_concept, "interval", minutes=6)
        print("--- [SCHEDULER]: Study Cycle is scheduled to run every 6 minutes. ---")

        # 2. The "Discovery" cycle runs infrequently to find brand new topics.
        scheduler.add_job(
            harvester.discover_new_topic_and_learn,
            "interval",
            minutes=35,
        )
        print(
            "--- [SCHEDULER]: Discovery Cycle is scheduled to run every 35 minutes. ---",
        )

        scheduler.start()

        print("\n--- [AUTONOMOUS TRAINER]: Agent is running in headless mode. ---")
        print("--- Knowledge Harvester is active. Press CTRL+C to stop. ---")

        # This is the main loop to keep the script alive.
        # The actual work is being done by the scheduler in background threads.
        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        print("\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Exiting. ---")
    except Exception as e:
        print(f"!!! [AUTONOMOUS TRAINER]: CRITICAL ERROR: {e} !!!")
        import traceback

        traceback.print_exc()
    finally:
        # In a real daemon, you might add cleanup here.
        print("--- [AUTONOMOUS TRAINER]: Process terminated. ---")


if __name__ == "__main__":
    start_autonomous_training()
