from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, Literal, TypedDict, cast

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent
    from .universal_interpreter import Task


logger = logging.getLogger(__name__)


class Goal(TypedDict, total=False):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    sub_goals: list[str | Task]
    stages: list[Goal]
    parent_goal: str | None
    priority: int


class GoalManager:
    """Manages the agent's high-level learning objectives with hierarchical planning."""

    def __init__(self, agent: CognitiveAgent):
        self.agent = agent
        self.goals: dict[str, Goal] = {}
        self._goal_counter = 0

    def _get_next_goal_id(self) -> str:
        """Generates a unique, sequential goal ID."""
        # Start goal IDs from 1 for clarity
        if not self.goals:
            self._goal_counter = 0

        # If loading from state, find the max existing goal number
        max_id = 0
        for goal_id in self.goals:
            if goal_id.startswith("goal_"):
                try:
                    num = int(goal_id.split("_")[1])
                    if num > max_id:
                        max_id = num
                except (ValueError, IndexError):
                    continue
        self._goal_counter = max_id + 1
        return f"goal_{self._goal_counter}"

    def _get_random_pedagogical_style(self) -> str:
        """Selects a random pedagogical style to ensure topic variety."""
        styles = [
            "a historian, focusing on origins and evolution.",
            "a practical engineer, focusing on application and function.",
            "a theoretical scientist, focusing on abstract principles.",
            "a teacher for a beginner, using the simplest possible terms.",
            "an encyclopedia author, providing a broad overview.",
            "a philosopher, focusing on first principles and definitions.",
        ]
        return random.choice(styles)

    def add_goal(
        self,
        description: str,
        parent_goal_id: str | None = None,
        priority: int = 5,
    ):
        """
        Add a new high-level learning goal and generate an appropriate study plan.
        """
        stage_pattern = re.compile(r"\[(Stage \d+:[^\]]+)\]:([^\[]+)")
        stages = stage_pattern.findall(description)

        if stages:
            main_goal_desc = description.split("[Stage")[0].strip()
            self._generate_staged_study_plan(
                main_goal_desc, stages, parent_goal_id, priority
            )
        else:
            self._generate_simple_study_plan(description, parent_goal_id, priority)

    def _generate_simple_study_plan(
        self, description: str, parent_goal_id: str | None, priority: int
    ):
        """Generates a flat list of sub-goals for a simple topic."""
        goal_id = self._get_next_goal_id()
        goal: Goal = {
            "id": goal_id,
            "description": description,
            "status": "pending",
            "sub_goals": [],
            "stages": [],
            "parent_goal": parent_goal_id,
            "priority": priority,
        }
        self.goals[goal_id] = goal
        logger.info("New simple goal added: '%s' (Priority: %d)", description, priority)

        topics = self.agent.interpreter.generate_curriculum(
            description, style=self._get_random_pedagogical_style()
        )
        if not topics:
            goal["status"] = "failed"
            logger.error("Failed to generate study plan for goal '%s'.", description)
            return

        sub_goals: list[str | Task] = [f"INVESTIGATE: {topic}" for topic in topics]
        goal["sub_goals"] = sub_goals
        goal["status"] = "in_progress"
        self.agent.learning_goals.extend(cast("list[str]", sub_goals))
        logger.info("Simple study plan generated with %d sub-goals.", len(sub_goals))

    def _generate_staged_study_plan(
        self,
        main_description: str,
        stages: list[tuple[str, str]],
        parent_goal_id: str | None,
        priority: int,
    ):
        """Generates a hierarchical, sequential plan for a multi-stage goal."""
        main_goal_id = self._get_next_goal_id()
        main_goal: Goal = {
            "id": main_goal_id,
            "description": main_description,
            # --- FIX: A parent goal's status is just 'pending' until all stages are complete ---
            "status": "pending",
            "sub_goals": [],
            "stages": [],
            "parent_goal": parent_goal_id,
            "priority": priority,
        }
        self.goals[main_goal_id] = main_goal
        logger.info(
            "[success][GoalManager]:New multi-stage goal added: '%s' with %d stages (Priority: %d).[/success]",
            main_description,
            len(stages),
            priority,
        )

        stage_goals: list[Goal] = []
        for i, (stage_name, stage_desc) in enumerate(stages):
            stage_id = self._get_next_goal_id()

            topics = self.agent.interpreter.generate_curriculum(
                stage_desc.strip(), style=self._get_random_pedagogical_style()
            )

            sub_goals: list[str | Task] = [f"INVESTIGATE: {topic}" for topic in topics]
            stage_goal: Goal = {
                "id": stage_id,
                "description": f"{stage_name}: {stage_desc.strip()}",
                "status": "pending" if i > 0 else "in_progress",
                "sub_goals": sub_goals,
                "stages": [],
                "parent_goal": main_goal_id,
                "priority": priority,
            }
            self.goals[stage_id] = stage_goal
            stage_goals.append(stage_goal)
            logger.info(
                "[purple]   - Generated plan for %s with %d concepts.[/purple]",
                stage_name,
                len(topics),
            )

        main_goal["stages"] = stage_goals
        if stage_goals:
            self.agent.learning_goals.extend(
                cast("list[str]", stage_goals[0]["sub_goals"])
            )
            logger.info(
                "[purple]Activating first stage with %d sub-goals.[/purple]",
                len(stage_goals[0]["sub_goals"]),
            )

    def get_active_goal(self) -> Goal | None:
        """
        Finds the single, unique, highest-priority goal or stage that is 'in_progress'.
        """
        # --- FIX: Simplify the logic to find the one true active goal ---
        in_progress_goals = [
            goal
            for goal in self.goals.values()
            if goal.get("status") == "in_progress"
            and goal.get("sub_goals")  # Only leaf-node goals can be active
        ]

        if not in_progress_goals:
            return None

        # If there's more than one, something is wrong, but we can recover by picking the highest priority one.
        if len(in_progress_goals) > 1:
            logger.warning(
                "Multiple 'in_progress' goals found. This may indicate a state error. Picking highest priority."
            )
            return sorted(
                in_progress_goals, key=lambda g: g.get("priority", 5), reverse=True
            )[0]

        return in_progress_goals[0]

    def check_goal_completion(self, goal_id: str):
        """Checks for completion of goals and activates the next stage if applicable."""
        goal = self.goals.get(goal_id)
        if not goal or goal.get("status") != "in_progress":
            return

        # A goal is complete if none of its sub_goals are in the main learning_goals queue.
        # We must normalize the sub-goals to just their topic strings for a fair comparison.
        sub_goal_topics = set()
        for sg in goal.get("sub_goals", []):
            if isinstance(sg, str):
                sub_goal_topics.add(sg.replace("INVESTIGATE:", "").strip())
            elif isinstance(sg, dict):
                sub_goal_topics.add(sg.get("topic", ""))

        agent_learning_goal_topics = set()
        for lg in self.agent.learning_goals:
            if isinstance(lg, str):
                agent_learning_goal_topics.add(lg.replace("INVESTIGATE:", "").strip())
            elif isinstance(lg, dict):
                agent_learning_goal_topics.add(lg.get("topic", ""))

        is_complete = not sub_goal_topics.intersection(agent_learning_goal_topics)

        if is_complete:
            goal["status"] = "completed"
            logger.info("🎉 Stage/Goal '%s' is complete! 🎉", goal.get("description"))

            if parent_goal_id := goal.get("parent_goal"):
                parent_goal = self.goals.get(parent_goal_id)
                if parent_goal and (stages := parent_goal.get("stages")):
                    completed_stage_index = -1
                    for i, stage in enumerate(stages):
                        if stage.get("id") == goal_id:
                            completed_stage_index = i
                            break

                    if 0 <= completed_stage_index < len(stages) - 1:
                        next_stage = stages[completed_stage_index + 1]
                        next_stage["status"] = "in_progress"
                        self.agent.learning_goals.extend(
                            cast("list[str]", next_stage.get("sub_goals", []))
                        )
                        logger.info(
                            "Activating next stage: '%s'", next_stage.get("description")
                        )
                    else:
                        # All stages are done, so the parent goal is now complete.
                        parent_goal["status"] = "completed"
                        logger.info(
                            "🎉 All stages for '%s' are complete. Main goal achieved! 🎉",
                            parent_goal.get("description"),
                        )
                        # --- FIX: Add the cleanup call here ---
                        self._clear_all_goals()
            else:
                # This handles the case of a simple, non-staged goal being completed.
                self._clear_all_goals()

    # --- FIX: Add the new cleanup method ---
    def _clear_all_goals(self) -> None:
        """
        Clears all goals from the manager, signaling the completion of the entire curriculum.
        """
        logger.info(
            "--- [GoalManager]: Entire curriculum complete. Resetting goal state. ---"
        )
        self.goals.clear()
        self._goal_counter = 0
