# tests/test_goal_manager.py


# We can directly use the 'agent' fixture from conftest.py,
# as it comes with a real GoalManager instance attached!


def test_get_active_goal_respects_priority(agent):
    """
    Tests that the goal with the highest priority number is returned as active.
    """
    # Arrange
    goal_manager = agent.goal_manager
    goal_manager.add_goal("Low priority task", priority=3)
    goal_manager.add_goal("High priority task", priority=8)
    goal_manager.add_goal("Medium priority task", priority=5)

    # Act
    active_goal = goal_manager.get_active_goal()

    # Assert
    assert active_goal is not None
    assert active_goal["description"] == "High priority task"
    assert active_goal["priority"] == 8
