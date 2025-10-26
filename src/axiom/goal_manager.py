from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class Goal(TypedDict):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    sub_goals: list[str]
    stages: list[Goal]
    parent_goal: str | None


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

    def add_goal(self, description: str):
        """
        Add a new high-level learning goal and generate an appropriate study plan.

        This method can parse both simple goals and complex, multi-stage goals.
        """
        stage_pattern = re.compile(r"\[(Stage \d+:[^\]]+)\]:([^\[]+)")
        stages = stage_pattern.findall(description)

        if stages:
            main_goal_desc = description.split("[Stage")[0].strip()
            self._generate_staged_study_plan(main_goal_desc, stages)
        else:
            self._generate_simple_study_plan(description)

    def _generate_simple_study_plan(self, description: str):
        """Generates a flat list of sub-goals for a simple topic."""
        goal_id = self._get_next_goal_id()
        goal: Goal = {
            "id": goal_id,
            "description": description,
            "status": "pending",
            "sub_goals": [],
            "stages": [],
            "parent_goal": None,
        }
        self.goals[goal_id] = goal
        logger.info("New simple goal added: '%s'", description)

        topics = self.agent.interpreter.generate_curriculum(description)
        if not topics:
            goal["status"] = "failed"
            logger.error("Failed to generate study plan for goal '%s'.", description)
            return

        sub_goals = [f"INVESTIGATE: {topic}" for topic in topics]
        goal["sub_goals"] = sub_goals
        goal["status"] = "in_progress"
        self.agent.learning_goals.extend(sub_goals)
        logger.info("Simple study plan generated with %d sub-goals.", len(sub_goals))

    def _generate_staged_study_plan(
        self, main_description: str, stages: list[tuple[str, str]]
    ):
        """Generates a hierarchical, sequential plan for a multi-stage goal."""
        main_goal_id = self._get_next_goal_id()
        main_goal: Goal = {
            "id": main_goal_id,
            "description": main_description,
            "status": "in_progress",
            "sub_goals": [],
            "stages": [],
            "parent_goal": None,
        }
        self.goals[main_goal_id] = main_goal
        logger.info("New multi-stage goal added: '%s'", main_description)

        stage_goals: list[Goal] = []
        for i, (stage_name, stage_desc) in enumerate(stages):
            stage_id = self._get_next_goal_id()
            stage_goal: Goal = {
                "id": stage_id,
                "description": f"{stage_name}: {stage_desc.strip()}",
                "status": "pending" if i > 0 else "in_progress",
                "sub_goals": [],
                "stages": [],
                "parent_goal": main_goal_id,
            }
            topics = self.agent.interpreter.generate_curriculum(stage_desc.strip())
            stage_goal["sub_goals"] = [f"INVESTIGATE: {topic}" for topic in topics]
            self.goals[stage_id] = stage_goal
            stage_goals.append(stage_goal)
            logger.info(
                "  - Generated plan for %s with %d concepts.", stage_name, len(topics)
            )

        main_goal["stages"] = stage_goals
        if stage_goals:
            self.agent.learning_goals.extend(stage_goals[0]["sub_goals"])
            logger.info(
                "Activating first stage with %d sub-goals.",
                len(stage_goals[0]["sub_goals"]),
            )

    def get_active_goal(self) -> Goal | None:
        """Finds the first goal or sub-goal stage that is 'in_progress'."""
        for goal in self.goals.values():
            if goal["status"] == "in_progress":
                if goal["stages"]:
                    for stage in goal["stages"]:
                        if stage["status"] == "in_progress":
                            return stage
                return goal
        return None

    def check_goal_completion(self, goal_id: str):
        """Checks for completion of goals and activates the next stage if applicable."""
        goal = self.goals.get(goal_id)
        if not goal or goal["status"] != "in_progress":
            return

        is_complete = not any(
            sg in self.agent.learning_goals for sg in goal["sub_goals"]
        )

        if is_complete:
            goal["status"] = "completed"
            logger.info("🎉 Stage/Goal '%s' is complete! 🎉", goal["description"])

            if goal["parent_goal"]:
                parent_goal = self.goals.get(goal["parent_goal"])
                if parent_goal:
                    completed_stage_index = -1
                    for i, stage in enumerate(parent_goal["stages"]):
                        if stage["id"] == goal_id:
                            completed_stage_index = i
                            break

                    if 0 <= completed_stage_index < len(parent_goal["stages"]) - 1:
                        next_stage = parent_goal["stages"][completed_stage_index + 1]
                        next_stage["status"] = "in_progress"
                        self.agent.learning_goals.extend(next_stage["sub_goals"])
                        logger.info(
                            "Activating next stage: '%s'", next_stage["description"]
                        )
                    else:
                        parent_goal["status"] = "completed"
                        logger.info(
                            "🎉 All stages for '%s' are complete. Main goal achieved! 🎉",
                            parent_goal["description"],
                        )
