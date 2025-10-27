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

        # TESTING DURATIONS
        self.LEARNING_PHASE_DURATION = timedelta(minutes=25)
        self.DISCOVERY_PHASE_DURATION = timedelta(minutes=10)
        self.REFINEMENT_PHASE_DURATION = timedelta(minutes=5)
        self.METACOGNITIVE_PHASE_DURATION = timedelta(minutes=9.5)

        self.current_phase: str | None = None
        self.phase_start_time: datetime | None = None

    def start(self) -> None:
        """Start the first phase and schedule all cognitive cycles."""
        self._start_learning_phase()

        # Check phases every 2 minutes for faster testing
        self.scheduler.add_job(
            self._manage_phases,
            "interval",
            minutes=2,
            id="cycle_manager_job",
        )

    def _manage_phases(self) -> None:
        """The main heartbeat, called every interval to check for phase transitions."""
        if not self.phase_start_time:
            return
        now = datetime.now()
        elapsed_time = now - self.phase_start_time
        if (
            self.current_phase == "learning"
            and elapsed_time >= self.LEARNING_PHASE_DURATION
        ):
            self._start_discovery_phase()
        elif (
            self.current_phase == "discovery"
            and elapsed_time >= self.DISCOVERY_PHASE_DURATION
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
            if job.id != "cycle_manager_job":
                job.remove()

    def _start_learning_phase(self) -> None:
        """Configure the scheduler to run the Learning Phase cycles."""
        logger.info("--- [CYCLE MANAGER]: Starting 5-minute LEARNING phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.harvester.study_cycle,
            "interval",
            minutes=1,
            id="study_cycle_job",
        )
        self.current_phase = "learning"
        self.phase_start_time = datetime.now()

    def _start_discovery_phase(self) -> None:
        """Configure the scheduler to run the Discovery Phase cycles."""
        logger.info("--- [CYCLE MANAGER]: Starting 5-minute DISCOVERY phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.harvester.discover_cycle,
            "interval",
            minutes=1,
            id="discover_cycle_job",
        )
        self.current_phase = "discovery"
        self.phase_start_time = datetime.now()

    def _start_refinement_phase(self) -> None:
        """Configure the scheduler to run the Refinement Phase cycles."""
        logger.info("--- [CYCLE MANAGER]: Starting 5-minute REFINEMENT phase. ---")
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
        logger.info("--- [CYCLE MANAGER]: Starting 5-minute METACOGNITIVE phase. ---")
        self._clear_all_jobs()
        self.scheduler.add_job(
            self.metacognitive_engine.run_introspection_cycle,
            "interval",
            minutes=5.9,
            id="metacognitive_cycle_job",
        )
        self.current_phase = "metacognitive"
        self.phase_start_time = datetime.now()
