from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from apscheduler.schedulers.base import BaseScheduler
    from rich.console import Console

    from ..knowledge_harvester import KnowledgeHarvester
    from ..metacognitive_engine import MetacognitiveEngine

logger = logging.getLogger("axiom")


class CycleManager:
    """
    Manages the agent's phased cognitive cycles using a responsive,
    event-driven, chained-execution model with a phase lock to prevent overlaps.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        harvester: KnowledgeHarvester,
        metacognitive_engine: MetacognitiveEngine,
        console: Console,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.harvester = harvester
        self.metacognitive_engine = metacognitive_engine
        self.console = console
        self.stop_event = stop_event
        self.phase_lock = threading.Lock()
        # Base durations (can be adjusted adaptively)
        self.BASE_LEARNING_PHASE_DURATION = timedelta(hours=1.2)
        self.BASE_DISCOVERY_PHASE_DURATION = timedelta(minutes=3.2)
        self.BASE_REFINEMENT_PHASE_DURATION = timedelta(minutes=3)
        self.BASE_METACOGNITIVE_PHASE_DURATION = timedelta(minutes=7)
        self.CYCLE_DELAY_SECONDS = 60
        self.current_phase: str | None = None
        self.phase_start_time: datetime | None = None
        # Adaptive thresholds
        self.PENDING_RELATIONS_HIGH_THRESHOLD = 50
        self.PHASE_EXTENSION_FACTOR = 1.2  # Extend by 20%
        fast_mode = os.getenv("AXIOM_FAST_TEST") == "1"
        if fast_mode:
            self.BASE_LEARNING_PHASE_DURATION = timedelta(minutes=5)
            self.BASE_DISCOVERY_PHASE_DURATION = timedelta(seconds=5)
            self.BASE_REFINEMENT_PHASE_DURATION = timedelta(minutes=5)
            self.BASE_METACOGNITIVE_PHASE_DURATION = timedelta(minutes=4)
            self.CYCLE_DELAY_SECONDS = 2

    def start(self, duration_hours: float | None = None) -> None:
        """
        Start the first phase, the high-level phase management job, and
        optionally schedule a shutdown job for benchmarking.
        """
        self._start_learning_phase()
        self.scheduler.add_job(
            self._manage_phases, "interval", minutes=1, id="cycle_manager_job"
        )
        if duration_hours is not None and duration_hours > 0:
            end_time = datetime.now() + timedelta(hours=duration_hours)

            def shutdown_after_delay():
                delay_seconds = (end_time - datetime.now()).total_seconds()
                if delay_seconds > 0:
                    timer = threading.Timer(delay_seconds, self.shutdown)
                    timer.start()
                    if self.stop_event:
                        # This will block the shutdown_thread until the main stop_event is set
                        self.stop_event.wait()
                        timer.cancel()

            shutdown_thread = threading.Thread(target=shutdown_after_delay, daemon=True)
            shutdown_thread.start()
            logger.warning(
                f"--- [CYCLE MANAGER]: Benchmark mode enabled. System will shut down at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---"
            )

    def shutdown(self, _wait: bool = True) -> None:
        """
        Cleanly shuts down the scheduler.
        """
        logger.info("--- [CYCLE MANAGER]: Shutdown initiated by job or signal. ---")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("--- [CYCLE MANAGER]: Scheduler has been shut down. ---")

    def _run_and_reschedule_cycle(
        self,
        cycle_func: Callable[..., bool],
        job_id: str,
        phase_name: str,
        **kwargs: object,
    ) -> None:
        """
        Generic wrapper to run a cycle function and reschedule it only if progress was made.
        """
        if self.stop_event and self.stop_event.is_set():
            logger.info(
                f"--- [CYCLE MANAGER]: Stop signal received. Skipping job '{job_id}'. ---"
            )
            return
        with self.phase_lock:
            progress_made = False
            try:
                progress_made = cycle_func(**kwargs)
            finally:
                if self.current_phase == phase_name and progress_made:
                    if self.stop_event and self.stop_event.is_set():
                        logger.info(
                            f"--- [CYCLE MANAGER]: Stop signal received. Not rescheduling '{job_id}'. ---"
                        )
                        return
                    next_run = datetime.now() + timedelta(
                        seconds=self.CYCLE_DELAY_SECONDS
                    )
                    self.scheduler.add_job(
                        self._run_and_reschedule_cycle,
                        "date",
                        run_date=next_run,
                        id=job_id,
                        args=[cycle_func, job_id, phase_name],
                        kwargs=kwargs,
                        max_instances=1,
                        misfire_grace_time=None,
                    )
                elif not progress_made:
                    logger.warning(
                        f"Cycle '{job_id}' reported no progress. Pausing execution until next phase."
                    )

    def _start_learning_phase(self) -> None:
        """Start the Learning Phase by scheduling the first study cycle."""
        # Adaptive duration based on pending relations
        agent = self.harvester.agent
        pending_count = len(agent.pending_relations)

        duration = self.BASE_LEARNING_PHASE_DURATION
        if pending_count > self.PENDING_RELATIONS_HIGH_THRESHOLD:
            duration = self.BASE_LEARNING_PHASE_DURATION * self.PHASE_EXTENSION_FACTOR
            logger.info(
                f"[Adaptive Phasing]: Extended learning phase by {int((self.PHASE_EXTENSION_FACTOR - 1) * 100)}% due to {pending_count} pending relations"
            )

        hours = duration.total_seconds() / 3600
        self.console.rule(
            f"[bold cyan]Starting {hours:.1f}-hour LEARNING phase (delay: {self.CYCLE_DELAY_SECONDS}s).[/bold cyan]",
            style="border",
        )
        self._clear_all_jobs()
        self.current_phase = "learning"
        self.phase_start_time = datetime.now()
        phase_end_time = self.phase_start_time + duration
        self.scheduler.add_job(
            self._run_and_reschedule_cycle,
            "date",
            run_date=datetime.now(),
            id="study_cycle_job",
            args=[self.harvester.study_cycle, "study_cycle_job", "learning"],
            kwargs={"phase_end_time": phase_end_time},
            max_instances=1,
        )

    def _start_discovery_phase(self) -> None:
        """Start the Discovery Phase by scheduling the first discover cycle."""
        minutes = self.BASE_DISCOVERY_PHASE_DURATION.total_seconds() / 60
        self.console.rule(
            f"[bold cyan]Starting {minutes:.1f}-minute DISCOVERY phase (delay: {self.CYCLE_DELAY_SECONDS}s).[/bold cyan]",
            style="border",
        )
        self._clear_all_jobs()
        self.current_phase = "discovery"
        self.phase_start_time = datetime.now()
        self.scheduler.add_job(
            self._run_and_reschedule_cycle,
            "date",
            run_date=datetime.now(),
            id="discover_cycle_job",
            args=[self.harvester.discover_cycle, "discover_cycle_job", "discovery"],
            max_instances=1,
        )

    def _start_refinement_phase(self) -> None:
        """Start the Refinement Phase by scheduling the first refinement cycle."""
        minutes = self.BASE_REFINEMENT_PHASE_DURATION.total_seconds() / 60
        self.console.rule(
            f"[bold blue]Starting {minutes:.1f}-minute REFINEMENT phase (delay: {self.CYCLE_DELAY_SECONDS}s).[/bold blue]",
            style="border",
        )
        self._clear_all_jobs()
        self.current_phase = "refinement"
        self.phase_start_time = datetime.now()
        self.scheduler.add_job(
            self._run_and_reschedule_cycle,
            "date",
            run_date=datetime.now(),
            id="refinement_cycle_job",
            args=[
                self.harvester.refinement_cycle,
                "refinement_cycle_job",
                "refinement",
            ],
            max_instances=1,
        )

    def _start_metacognitive_phase(self) -> None:
        """Start the Metacognitive Phase by scheduling the first metacognitive cycle."""
        minutes = self.BASE_METACOGNITIVE_PHASE_DURATION.total_seconds() / 60
        self.console.rule(
            f"[bold blue]Starting {minutes:.1f}-minute METACOGNITIVE phase (delay: {self.CYCLE_DELAY_SECONDS}s).[/bold blue]",
            style="border",
        )
        self._clear_all_jobs()
        self.current_phase = "metacognitive"
        self.phase_start_time = datetime.now()
        self.scheduler.add_job(
            self._run_and_reschedule_cycle,
            "date",
            run_date=datetime.now(),
            id="metacognitive_cycle_job",
            args=[
                self.metacognitive_engine.run_introspection_cycle,
                "metacognitive_cycle_job",
                "metacognitive",
            ],
            max_instances=1,
        )

    def _manage_phases(self) -> None:
        """The main heartbeat, called every interval to check for phase transitions."""
        # --- FIX: Check the stop event at the very beginning of the heartbeat ---
        if self.stop_event and self.stop_event.is_set():
            logger.info(
                "--- [CYCLE MANAGER]: Stop signal detected in heartbeat. Shutting down scheduler. ---"
            )
            self.shutdown()
            return

        with self.phase_lock:
            if not self.phase_start_time:
                return
            now = datetime.now()
            elapsed_time = now - self.phase_start_time
            # Log metrics before transitioning
            if self.current_phase:
                self._log_phase_metrics(self.current_phase)

            # Calculate adaptive duration for learning phase
            learning_duration = self.BASE_LEARNING_PHASE_DURATION
            agent = self.harvester.agent
            if len(agent.pending_relations) > self.PENDING_RELATIONS_HIGH_THRESHOLD:
                learning_duration = (
                    self.BASE_LEARNING_PHASE_DURATION * self.PHASE_EXTENSION_FACTOR
                )

            if self.current_phase == "learning" and elapsed_time >= learning_duration:
                self._start_discovery_phase()
            elif (
                self.current_phase == "discovery"
                and elapsed_time >= self.BASE_DISCOVERY_PHASE_DURATION
            ):
                self._start_refinement_phase()
            elif (
                self.current_phase == "refinement"
                and elapsed_time >= self.BASE_REFINEMENT_PHASE_DURATION
            ):
                self._start_metacognitive_phase()
            elif (
                self.current_phase == "metacognitive"
                and elapsed_time >= self.BASE_METACOGNITIVE_PHASE_DURATION
            ):
                self._start_learning_phase()

    def _log_phase_metrics(self, phase_name: str) -> None:
        """Log comprehensive metrics at the end of a phase."""
        agent = self.harvester.agent
        node_count = len(agent.graph.get_all_nodes())
        edge_count = len(agent.graph.get_all_edges())
        pending_count = len(agent.pending_relations)
        goal_count = len(agent.learning_goals) + len(agent.high_priority_tasks)

        logger.info(
            f"[Phase Metrics - {phase_name.upper()}]: "
            f"Nodes={node_count}, Edges={edge_count}, "
            f"Pending={pending_count}, Active Goals={goal_count}"
        )

    def _clear_all_jobs(self) -> None:
        """Remove all existing cognitive cycle jobs from the scheduler."""
        for job in self.scheduler.get_jobs():
            if job.id not in ["cycle_manager_job", "shutdown_job"]:
                job.remove()
