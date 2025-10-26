import threading
from unittest.mock import MagicMock

import pytest

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester


@pytest.fixture
def harvester(agent: CognitiveAgent) -> KnowledgeHarvester:
    """Creates a KnowledgeHarvester instance linked to the test agent."""
    lock = threading.Lock()
    return KnowledgeHarvester(agent, lock)


def test_discover_cycle_adds_goal(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch
):
    """
    Given: The internal _find_new_topic method returns a valid topic.
    When: The discover_cycle is executed.
    Then: A new 'INVESTIGATE' goal is added to the agent's learning goals.
    """
    monkeypatch.setattr(harvester, "_find_new_topic", lambda: "Quantum Mechanics")

    agent.learning_goals.clear()
    harvester.discover_cycle()

    assert "INVESTIGATE: Quantum Mechanics" in agent.learning_goals


# --- THIS IS THE UPDATED TEST ---
def test_resolve_investigation_goal_with_api_success(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch
):
    """
    Given: The dictionary API returns a valid definition.
    When: _resolve_investigation_goal is called.
    Then: It should learn the part-of-speech and pass the definition sentence
          to the autonomous learner, then remove the goal.
    """
    # 1. Mock external dependencies
    monkeypatch.setattr(
        harvester,
        "get_definition_from_api",
        lambda word: ("noun", "A test definition."),
    )

    # 2. Spy on the internal functions that are now called
    # We patch the imported 'validate_and_add_relation' function within the harvester's namespace
    validate_spy = MagicMock(return_value="inserted")
    monkeypatch.setattr(
        "axiom.knowledge_harvester.validate_and_add_relation", validate_spy
    )

    autonomous_learn_spy = MagicMock(return_value=True)
    monkeypatch.setattr(agent, "learn_new_fact_autonomously", autonomous_learn_spy)

    # 3. Set up the initial state
    goal = "INVESTIGATE: testword"
    agent.learning_goals.append(goal)

    # 4. Run the function under test
    resolved = harvester._resolve_investigation_goal(goal)

    # 5. Assert the results
    assert resolved is True
    assert goal not in agent.learning_goals

    # Assert that it tried to learn the part-of-speech
    validate_spy.assert_called_once()
    # Inspect the arguments passed to the validation function
    pos_relation_arg = validate_spy.call_args[0][1]
    assert pos_relation_arg == {
        "subject": "testword",
        "verb": "is_a",
        "object": "noun",
        "properties": {"provenance": "dictionary_api"},
    }

    # Assert that it passed the definition sentence to the autonomous learner
    autonomous_learn_spy.assert_called_once_with(
        fact_sentence="A test definition.",
        source_topic="testword",
    )


def test_resolve_investigation_goal_falls_back_to_web(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch
):
    """
    Given: The dictionary API fails.
    When: _resolve_investigation_goal is called.
    Then: It should fall back to web search, learn successfully, and remove the goal.
    """
    monkeypatch.setattr(harvester, "get_definition_from_api", lambda word: None)
    monkeypatch.setattr(
        harvester,
        "get_fact_from_wikipedia",
        lambda topic: ("A Test Topic", "The test topic is a noun."),
    )
    monkeypatch.setattr(harvester, "get_fact_from_duckduckgo", lambda topic: None)

    learn_spy = MagicMock(return_value=True)
    monkeypatch.setattr(agent, "learn_new_fact_autonomously", learn_spy)

    goal = "INVESTIGATE: fallbackword"
    agent.learning_goals.append(goal)

    resolved = harvester._resolve_investigation_goal(goal)

    assert resolved is True
    assert goal not in agent.learning_goals
    learn_spy.assert_called_once_with(
        fact_sentence="The test topic is a noun.", source_topic="A Test Topic"
    )


def test_resolve_investigation_goal_handles_total_failure(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch
):
    """
    Given: All external knowledge sources fail.
    When: _resolve_investigation_goal is called.
    Then: It should return False and NOT remove the goal.
    """
    monkeypatch.setattr(harvester, "get_definition_from_api", lambda word: None)
    monkeypatch.setattr(harvester, "get_fact_from_wikipedia", lambda topic: None)
    monkeypatch.setattr(harvester, "get_fact_from_duckduckgo", lambda topic: None)

    goal = "INVESTIGATE: failureword"
    agent.learning_goals.append(goal)

    resolved = harvester._resolve_investigation_goal(goal)

    assert resolved is False
    assert goal in agent.learning_goals


def test_study_cycle_removes_failed_opportunistic_goal(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch
):
    """
    Given: There is no active plan and an opportunistic goal fails to resolve.
    When: The study_cycle runs.
    Then: The failed goal is removed from the learning queue to prevent loops.
    """
    monkeypatch.setattr(harvester, "_resolve_investigation_goal", lambda goal: False)
    monkeypatch.setattr(agent.goal_manager, "get_active_goal", lambda: None)

    goal = "INVESTIGATE: someword"
    agent.learning_goals.append(goal)

    harvester.study_cycle()

    assert goal not in agent.learning_goals


def test_study_cycle_prioritizes_planned_goal(
    harvester: KnowledgeHarvester, agent: CognitiveAgent, monkeypatch
):
    """
    Given: An active plan exists in the GoalManager.
    When: The study_cycle runs.
    Then: It attempts to resolve a task from the active plan, not an opportunistic one.
    """
    active_goal = {
        "id": "goal_1",
        "description": "Active Plan",
        "status": "in_progress",
        "sub_goals": ["INVESTIGATE: planned_task"],
        "stages": [],
        "parent_goal": None,
    }
    monkeypatch.setattr(agent.goal_manager, "get_active_goal", lambda: active_goal)

    agent.learning_goals.clear()
    agent.learning_goals.append("INVESTIGATE: opportunistic_task")
    agent.learning_goals.append("INVESTIGATE: planned_task")

    resolve_spy = MagicMock(return_value=True)
    monkeypatch.setattr(harvester, "_resolve_investigation_goal", resolve_spy)

    completion_spy = MagicMock()
    monkeypatch.setattr(agent.goal_manager, "check_goal_completion", completion_spy)

    harvester.study_cycle()

    resolve_spy.assert_called_once_with("INVESTIGATE: planned_task")
    completion_spy.assert_called_once_with("goal_1")
