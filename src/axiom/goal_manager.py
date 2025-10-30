from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, Literal, TypedDict, cast

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class Goal(TypedDict, total=False):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    sub_goals: list[str | Goal]
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
        self._goal_counter += 1
        return f"goal_{self._goal_counter}"

    # --- NEW: Helper method to provide a random creative seed ---
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

        # FIX: Inject a random style to ensure variety.
        topics = self.agent.interpreter.generate_curriculum(
            description, style=self._get_random_pedagogical_style()
        )
        if not topics:
            goal["status"] = "failed"
            logger.error("Failed to generate study plan for goal '%s'.", description)
            return

        sub_goals: list[str | Goal] = [f"INVESTIGATE: {topic}" for topic in topics]
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
            "status": "in_progress",
            "sub_goals": [],
            "stages": [],
            "parent_goal": parent_goal_id,
            "priority": priority,
        }
        self.goals[main_goal_id] = main_goal
        logger.info(
            "New multi-stage goal added: '%s' with %d stages (Priority: %d).",
            main_description,
            len(stages),
            priority,
        )

        stage_goals: list[Goal] = []
        for i, (stage_name, stage_desc) in enumerate(stages):
            stage_id = self._get_next_goal_id()

            # FIX: Inject a random style for each stage to maximize variety.
            topics = self.agent.interpreter.generate_curriculum(
                stage_desc.strip(), style=self._get_random_pedagogical_style()
            )

            sub_goals: list[str | Goal] = [f"INVESTIGATE: {topic}" for topic in topics]
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
                "  - Generated plan for %s with %d concepts.", stage_name, len(topics)
            )

        main_goal["stages"] = stage_goals
        if stage_goals:
            self.agent.learning_goals.extend(
                cast("list[str]", stage_goals[0]["sub_goals"])
            )
            logger.info(
                "Activating first stage with %d sub-goals.",
                len(stage_goals[0]["sub_goals"]),
            )

    def get_active_goal(self) -> Goal | None:
        """
        Finds the highest-priority, 'in_progress' goal or stage.
        """
        in_progress_goals = [
            goal for goal in self.goals.values() if goal.get("status") == "in_progress"
        ]
        if not in_progress_goals:
            return None

        highest_priority_goal = sorted(
            in_progress_goals, key=lambda g: g.get("priority", 5), reverse=True
        )[0]

        if stages := highest_priority_goal.get("stages"):
            for stage in stages:
                if stage.get("status") == "in_progress":
                    return stage

        return highest_priority_goal

    def check_goal_completion(self, goal_id: str):
        """Checks for completion of goals and activates the next stage if applicable."""
        goal = self.goals.get(goal_id)
        if not goal or goal.get("status") != "in_progress":
            return

        sub_goals_as_str = [
            sg for sg in goal.get("sub_goals", []) if isinstance(sg, str)
        ]
        is_complete = not any(
            sg in self.agent.learning_goals for sg in sub_goals_as_str
        )

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
                        parent_goal["status"] = "completed"
                        logger.info(
                            "🎉 All stages for '%s' are complete. Main goal achieved! 🎉",
                            parent_goal.get("description"),
                        )
