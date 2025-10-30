from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apscheduler.schedulers.base import BaseScheduler

    from ..knowledge_harvester import KnowledgeHarvester
    from ..metacognitive_engine import MetacognitiveEngine

logger = logging.getLogger(__name__)


class CycleManager:
    """
    Manages the agent's phased cognitive cycles using a responsive,
    event-driven, chained-execution model.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        harvester: KnowledgeHarvester,
        metacognitive_engine: MetacognitiveEngine,
    ) -> None:
        self.scheduler = scheduler
        self.harvester = harvester
        self.metacognitive_engine = metacognitive_engine

        # Phase durations
        self.LEARNING_PHASE_DURATION = timedelta(minutes=45)
        self.DISCOVERY_PHASE_DURATION = timedelta(minutes=5)
        self.REFINEMENT_PHASE_DURATION = timedelta(minutes=5)
        self.METACOGNITIVE_PHASE_DURATION = timedelta(minutes=5)

        # --- NEW: Configurable delay between cycle completions ---
        self.CYCLE_DELAY_SECONDS = 15

        self.current_phase: str | None = None
        self.phase_start_time: datetime | None = None

        # --- NEW: Configurable delay for testing ---
        fast_mode = os.getenv("AXIOM_FAST_TEST") == "1"
        if fast_mode:
            # super short durations for testing
            self.LEARNING_PHASE_DURATION = timedelta(seconds=5)
            self.DISCOVERY_PHASE_DURATION = timedelta(seconds=5)
            self.REFINEMENT_PHASE_DURATION = timedelta(seconds=5)
            self.METACOGNITIVE_PHASE_DURATION = timedelta(seconds=5)
            self.CYCLE_DELAY_SECONDS = 2

    def start(self) -> None:
        """Start the first phase and the high-level phase management job."""
        self._start_learning_phase()

        # The high-level phase manager still runs on a fixed interval.
        self.scheduler.add_job(
            self._manage_phases,
            "interval",
            minutes=1,
            id="cycle_manager_job",
        )

    # --- NEW: Wrapper methods for each cycle to enable self-rescheduling ---

    def _run_study_cycle(self) -> None:
        """Wrapper to run the study cycle and schedule the next one."""
        try:
            self.harvester.study_cycle()
        finally:
            # If we are still in the learning phase, schedule the next run.
            if self.current_phase == "learning":
                next_run = datetime.now() + timedelta(seconds=self.CYCLE_DELAY_SECONDS)
                self.scheduler.add_job(
                    self._run_study_cycle,
                    "date",
                    run_date=next_run,
                    id="study_cycle_job",
                    max_instances=1,
                    misfire_grace_time=None,  # Run immediately if missed
                )

    def _run_discover_cycle(self) -> None:
        """Wrapper to run the discover cycle and schedule the next one."""
        try:
            self.harvester.discover_cycle()
        finally:
            if self.current_phase == "discovery":
                next_run = datetime.now() + timedelta(seconds=self.CYCLE_DELAY_SECONDS)
                self.scheduler.add_job(
                    self._run_discover_cycle,
                    "date",
                    run_date=next_run,
                    id="discover_cycle_job",
                    max_instances=1,
                    misfire_grace_time=None,
                )

    def _run_refinement_cycle(self) -> None:
        """Wrapper to run the refinement cycle and schedule the next one."""
        try:
            self.harvester.refinement_cycle()
        finally:
            if self.current_phase == "refinement":
                next_run = datetime.now() + timedelta(seconds=self.CYCLE_DELAY_SECONDS)
                self.scheduler.add_job(
                    self._run_refinement_cycle,
                    "date",
                    run_date=next_run,
                    id="refinement_cycle_job",
                    max_instances=1,
                    misfire_grace_time=None,
                )

    def _run_metacognitive_cycle(self) -> None:
        """Wrapper to run the metacognitive cycle and schedule the next one."""
        try:
            self.metacognitive_engine.run_introspection_cycle()
        finally:
            if self.current_phase == "metacognitive":
                next_run = datetime.now() + timedelta(seconds=self.CYCLE_DELAY_SECONDS)
                self.scheduler.add_job(
                    self._run_metacognitive_cycle,
                    "date",
                    run_date=next_run,
                    id="metacognitive_cycle_job",
                    max_instances=1,
                    misfire_grace_time=None,
                )

    # --- MODIFIED: Phase start methods now kick off the first job in the chain ---

    def _start_learning_phase(self) -> None:
        """Start the Learning Phase by scheduling the first study cycle."""
        logger.info(
            "--- [CYCLE MANAGER]: Starting %.1f-hour LEARNING phase (delay: %ds). ---",
            self.LEARNING_PHASE_DURATION.total_seconds() / 3600,
            self.CYCLE_DELAY_SECONDS,
        )
        self._clear_all_jobs()
        self.current_phase = "learning"
        self.phase_start_time = datetime.now()
        # Schedule the very first run to happen immediately.
        self.scheduler.add_job(
            self._run_study_cycle,
            "date",
            run_date=datetime.now(),
            id="study_cycle_job",
            max_instances=1,
        )

    def _start_discovery_phase(self) -> None:
        """Start the Discovery Phase by scheduling the first discover cycle."""
        logger.info(
            "--- [CYCLE MANAGER]: Starting %.1f-minute DISCOVERY phase (delay: %ds). ---",
            self.DISCOVERY_PHASE_DURATION.total_seconds() / 60,
            self.CYCLE_DELAY_SECONDS,
        )
        self._clear_all_jobs()
        self.current_phase = "discovery"
        self.phase_start_time = datetime.now()
        self.scheduler.add_job(
            self._run_discover_cycle,
            "date",
            run_date=datetime.now(),
            id="discover_cycle_job",
            max_instances=1,
        )

    def _start_refinement_phase(self) -> None:
        """Start the Refinement Phase by scheduling the first refinement cycle."""
        logger.info(
            "--- [CYCLE MANAGER]: Starting %.1f-minute REFINEMENT phase (delay: %ds). ---",
            self.REFINEMENT_PHASE_DURATION.total_seconds() / 60,
            self.CYCLE_DELAY_SECONDS,
        )
        self._clear_all_jobs()
        self.current_phase = "refinement"
        self.phase_start_time = datetime.now()
        self.scheduler.add_job(
            self._run_refinement_cycle,
            "date",
            run_date=datetime.now(),
            id="refinement_cycle_job",
            max_instances=1,
        )

    def _start_metacognitive_phase(self) -> None:
        """Start the Metacognitive Phase by scheduling the first metacognitive cycle."""
        logger.info(
            "--- [CYCLE MANAGER]: Starting %.1f-minute METACOGNITIVE phase (delay: %ds). ---",
            self.METACOGNITIVE_PHASE_DURATION.total_seconds() / 60,
            self.CYCLE_DELAY_SECONDS,
        )
        self._clear_all_jobs()
        self.current_phase = "metacognitive"
        self.phase_start_time = datetime.now()
        self.scheduler.add_job(
            self._run_metacognitive_cycle,
            "date",
            run_date=datetime.now(),
            id="metacognitive_cycle_job",
            max_instances=1,
        )

    # --- Unchanged Helper Methods ---

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
