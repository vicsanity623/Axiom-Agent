from __future__ import annotations

# autonomous_trainer.py
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from apscheduler.schedulers.background import BackgroundScheduler

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester

if TYPE_CHECKING:
    from apscheduler.schedulers.base import BaseScheduler

BRAIN_PATH: Final = Path("brain")
BRAIN_FILE: Final = BRAIN_PATH / "my_agent_brain.json"
STATE_FILE: Final = BRAIN_PATH / "my_agent_state.json"


class CycleManager:
    """Manages the agent's phased cognitive cycles (learning vs. refinement)."""

    def __init__(self, scheduler: BaseScheduler, harvester: KnowledgeHarvester) -> None:
        self.scheduler = scheduler
        self.harvester = harvester
        self.LEARNING_PHASE_DURATION = timedelta(hours=4)
        self.REFINEMENT_PHASE_DURATION = timedelta(hours=1)
        self.current_phase: str | None = None
        self.phase_start_time: datetime | None = None

    def start(self) -> None:
        """Start the first phase and schedule the manager's main loop."""
        self._start_learning_phase()
        self.scheduler.add_job(
            self._manage_phases,
            "interval",
            minutes=1,
            id="cycle_manager_job",
        )

    def _manage_phases(self) -> None:
        """The main heartbeat, called every minute to check for phase transitions."""
        if not self.phase_start_time:
            return

        now = datetime.now()
        elapsed_time = now - self.phase_start_time

        if (
            self.current_phase == "learning"
            and elapsed_time >= self.LEARNING_PHASE_DURATION
        ):
            self._start_refinement_phase()
        elif (
            self.current_phase == "refinement"
            and elapsed_time >= self.REFINEMENT_PHASE_DURATION
        ):
            self._start_learning_phase()

    def _clear_all_jobs(self) -> None:
        """Remove all existing cognitive cycle jobs from the scheduler."""
        for job in self.scheduler.get_jobs():
            if job.id in [
                "study_cycle_job",
                "discover_cycle_job",
                "refinement_cycle_job",
            ]:
                job.remove()

    def _start_learning_phase(self) -> None:
        """Configure the scheduler to run the Learning Phase cycles."""
        print("\n--- [CYCLE MANAGER]: Starting 4-hour LEARNING phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.harvester.study_cycle,
            "interval",
            minutes=6,
            id="study_cycle_job",
        )
        self.scheduler.add_job(
            self.harvester.discover_cycle,
            "interval",
            minutes=21,
            id="discover_cycle_job",
        )
        self.current_phase = "learning"
        self.phase_start_time = datetime.now()

    def _start_refinement_phase(self) -> None:
        """Configure the scheduler to run the Refinement Phase cycles."""
        print("\n--- [CYCLE MANAGER]: Starting 1-hour REFINEMENT phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.harvester.refinement_cycle,
            "interval",
            minutes=6,
            id="refinement_cycle_job",
        )
        self.current_phase = "refinement"
        self.phase_start_time = datetime.now()


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


if __name__ == "__main__":
    start_autonomous_training(BRAIN_FILE, STATE_FILE)
