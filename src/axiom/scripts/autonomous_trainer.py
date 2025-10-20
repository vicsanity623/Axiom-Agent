from __future__ import annotations

import logging

# autonomous_trainer.py
import threading
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE
from ..knowledge_harvester import KnowledgeHarvester
from .cycle_manager import CycleManager


def start_autonomous_training(brain_file: Path, state_file: Path) -> None:
    """Initialize the agent and run its autonomous learning cycles indefinitely."""
    print("--- [AUTONOMOUS TRAINER]: Starting Axiom Agent Initialization... ---")
    agent_interaction_lock = threading.Lock()

    try:
        axiom_agent = CognitiveAgent(
            brain_file=brain_file,
            state_file=state_file,
            inference_mode=False,
        )

        harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)
        scheduler = BackgroundScheduler(daemon=True)

        manager = CycleManager(scheduler, harvester)
        manager.start()

        scheduler.start()

        print("\n--- [AUTONOMOUS TRAINER]: Agent is running in headless mode. ---")
        print("--- Knowledge Harvester is active. Press CTRL+C to stop. ---")

        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        print("\n--- [AUTONOMOUS TRAINER]: Shutdown signal received. Exiting. ---")
    except Exception as exc:
        print(f"!!! [AUTONOMOUS TRAINER]: CRITICAL ERROR: {exc} !!!")
        traceback.print_exc()
    finally:
        print("--- [AUTONOMOUS TRAINER]: Process terminated. ---")


def main() -> None:
    """Entry point for the axiom-train command."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    start_autonomous_training(DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE)


if __name__ == "__main__":
    main()
