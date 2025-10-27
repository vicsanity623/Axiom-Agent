from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apscheduler.schedulers.base import BaseScheduler

    from ..knowledge_harvester import KnowledgeHarvester
    from ..metacognitive_engine import MetacognitiveEngine

logger = logging.getLogger(__name__)


class CycleManager:
    """Manages the agent's phased cognitive cycles (learning vs. refinement)."""

    def __init__(
        self,
        scheduler: BaseScheduler,
        harvester: KnowledgeHarvester,
        metacognitive_engine: MetacognitiveEngine,
    ) -> None:
        self.scheduler = scheduler
        self.harvester = harvester
        self.metacognitive_engine = metacognitive_engine
        self.LEARNING_PHASE_DURATION = timedelta(hours=2)
        self.REFINEMENT_PHASE_DURATION = timedelta(minutes=45)
        self.METACOGNITIVE_PHASE_DURATION = timedelta(minutes=9)
        self.current_phase: str | None = None
        self.phase_start_time: datetime | None = None

    def start(self) -> None:
        """Start the first phase and schedule all cognitive cycles."""
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
            self._start_metacognitive_phase()
        elif (
            self.current_phase == "metacognitive"
            and elapsed_time >= self.METACOGNITIVE_PHASE_DURATION
        ):
            self._start_learning_phase()

    def _clear_all_jobs(self) -> None:
        """Remove all existing cognitive cycle jobs from the scheduler."""
        for job in self.scheduler.get_jobs():
            if job.id not in ["cycle_manager_job"]:
                job.remove()

    def _start_learning_phase(self) -> None:
        """Configure the scheduler to run the Learning Phase cycles."""
        logger.info("--- [CYCLE MANAGER]: Starting 3-hour LEARNING phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.harvester.study_cycle,
            "interval",
            minutes=5.5,
            id="study_cycle_job",
        )
        self.scheduler.add_job(
            self.harvester.discover_cycle,
            "interval",
            minutes=20,
            id="discover_cycle_job",
        )
        self.current_phase = "learning"
        self.phase_start_time = datetime.now()

    def _start_refinement_phase(self) -> None:
        """Configure the scheduler to run the Refinement Phase cycles."""
        logger.info("--- [CYCLE MANAGER]: Starting 1-hour REFINEMENT phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.harvester.refinement_cycle,
            "interval",
            minutes=9,
            id="refinement_cycle_job",
        )

        self.current_phase = "refinement"
        self.phase_start_time = datetime.now()

    def _start_metacognitive_phase(self) -> None:
        """Configure the scheduler to run the Metacognitive Phase cycles."""
        logger.info("--- [CYCLE MANAGER]: Starting 1-hour METACOGNITIVE phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.metacognitive_engine.run_introspection_cycle,
            "interval",
            minutes=5.2,
            id="metacognitive_cycle_job",
        )

        self.current_phase = "metacognitive"
        self.phase_start_time = datetime.now()
