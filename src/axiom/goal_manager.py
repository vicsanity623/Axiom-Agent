from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, TypedDict  # Add TYPE_CHECKING

if TYPE_CHECKING:
    from .cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)


class Goal(TypedDict):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    sub_goals: list[str]


class GoalManager:
    """Manages the agent's high-level learning objectives."""

    def __init__(self, agent: CognitiveAgent):
        self.agent = agent
        self.goals: dict[str, Goal] = {}

    def add_goal(self, description: str):
        """Adds a new high-level learning goal."""
        goal_id = f"goal_{len(self.goals) + 1}"
        self.goals[goal_id] = {
            "id": goal_id,
            "description": description,
            "status": "pending",
            "sub_goals": [],
        }
        logger.info("New high-level goal added: '%s'", description)
        self.generate_study_plan(goal_id)

    def get_active_goal(self) -> Goal | None:
        """Finds the first goal that is 'in_progress'."""
        for goal in self.goals.values():
            if goal["status"] == "in_progress":
                return goal
        return None

    def generate_study_plan(self, goal_id: str):
        """Uses the LLM to break a high-level goal into a curriculum of study topics."""
        goal = self.goals.get(goal_id)
        if not goal:
            return

        logger.info(
            "Generating strategic study plan for goal: '%s'",
            goal["description"],
        )

        topics_to_investigate = self.agent.interpreter.generate_curriculum(
            goal["description"],
        )

        if not topics_to_investigate:
            goal["status"] = "failed"
            logger.error(
                "Failed to generate study plan for goal '%s'.",
                goal["description"],
            )
            return

        sub_goals = [f"INVESTIGATE: {topic}" for topic in topics_to_investigate]
        goal["sub_goals"] = sub_goals
        goal["status"] = "in_progress"

        self.agent.learning_goals.extend(sub_goals)
        logger.info(
            "Study plan generated with %d sub-goals. Added to learning queue.",
            len(sub_goals),
        )

    def check_goal_completion(self, goal_id: str):
        """Checks if all sub-goals for a given goal have been resolved."""
        goal = self.goals.get(goal_id)
        if not goal or goal["status"] != "in_progress":
            return

        is_complete = not any(
            sub_goal in self.agent.learning_goals for sub_goal in goal["sub_goals"]
        )

        if is_complete:
            goal["status"] = "completed"
            logger.info(
                "🎉 All sub-goals for '%s' are complete. Goal achieved! 🎉",
                goal["description"],
            )
