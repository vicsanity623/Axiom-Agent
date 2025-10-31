# tests/scripts/test_autonomous_trainer.py

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the functions we want to test
from axiom.scripts.autonomous_trainer import (
    _generate_or_set_initial_goal,
    main,
    start_autonomous_training,
)

# Use a common prefix for patching to make it cleaner
MODULE_PATH = "axiom.scripts.autonomous_trainer"


@pytest.fixture
def mock_app_components(mocker, agent):
    """
    A fixture that prepares the application environment for testing.

    It leverages the `agent` fixture from conftest.py and mocks the
    other components that `start_autonomous_training` creates.
    """
    mocker.patch(f"{MODULE_PATH}.CognitiveAgent", return_value=agent)

    mock_engine_class = mocker.patch(f"{MODULE_PATH}.MetacognitiveEngine")
    mock_manager_class = mocker.patch(f"{MODULE_PATH}.CycleManager")
    mock_scheduler_class = mocker.patch(f"{MODULE_PATH}.BackgroundScheduler")
    mock_console = mocker.patch(f"{MODULE_PATH}.console")
    mock_logger = mocker.patch(f"{MODULE_PATH}.logger")

    mock_sleep = mocker.patch(f"{MODULE_PATH}.time.sleep")
    mock_sleep.side_effect = KeyboardInterrupt("Test Shutdown")

    mock_manager_instance = mock_manager_class.return_value
    mock_scheduler_instance = mock_scheduler_class.return_value
    mock_engine_class = mocker.patch(f"{MODULE_PATH}.MetacognitiveEngine")

    return {
        "agent": agent,
        "manager_instance": mock_manager_instance,
        "MetacognitiveEngine": mock_engine_class,
        "scheduler_instance": mock_scheduler_instance,
        "console": mock_console,
        "logger": mock_logger,
        "sleep": mock_sleep,
    }


def test_start_autonomous_training_generates_new_goal(mock_app_components, tmp_path):
    """
    Tests the main success path: a new agent generates a new curriculum.
    """
    agent = mock_app_components["agent"]
    assert agent.goal_manager.get_active_goal() is None

    start_autonomous_training(tmp_path / "b.json", tmp_path / "s.json")

    active_goal = agent.goal_manager.get_active_goal()
    assert active_goal is not None
    assert "Stage 1:" in active_goal["description"]

    mock_app_components["MetacognitiveEngine"].assert_called_once()
    mock_app_components["manager_instance"].start.assert_called_once()
    mock_app_components["scheduler_instance"].start.assert_called_once()
    mock_app_components["sleep"].assert_called_once()


def test_start_autonomous_training_uses_custom_goal(mock_app_components, tmp_path):
    """Tests that a provided initial_goal is used directly."""
    agent = mock_app_components["agent"]
    custom_goal = "My specific learning objective."

    start_autonomous_training(
        tmp_path / "b.json", tmp_path / "s.json", initial_goal=custom_goal
    )

    active_goal = agent.goal_manager.get_active_goal()
    assert active_goal is not None
    assert active_goal["description"] == custom_goal
    assert len(agent.goal_manager.goals) == 1


def test_start_autonomous_training_resumes_with_existing_goal(
    mock_app_components, tmp_path
):
    """Tests that if an agent already has a goal, it's not replaced."""
    agent = mock_app_components["agent"]
    existing_goal = "Continue learning about birds."
    agent.goal_manager.add_goal(existing_goal)

    start_autonomous_training(tmp_path / "b.json", tmp_path / "s.json")

    active_goal = agent.goal_manager.get_active_goal()
    assert active_goal is not None
    assert active_goal["description"] == existing_goal
    assert len(agent.goal_manager.goals) == 1


def test_generate_goal_handles_failed_generation(mock_app_components):
    """
    Tests that if the mock LLM fails, no goal is added.
    We can achieve this by monkeypatching the mock interpreter for one test.
    """
    agent = mock_app_components["agent"]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(agent.interpreter, "synthesize", lambda *args, **kwargs: None)

    _generate_or_set_initial_goal(agent, None)

    assert agent.goal_manager.get_active_goal() is None
    mock_app_components["logger"].error.assert_called_once()
    monkeypatch.undo()


def test_start_autonomous_training_handles_critical_error(
    mock_app_components, tmp_path, mocker
):
    """Tests the main exception handling block by simulating an init failure."""
    error_message = "Disk is full"
    mocker.patch(f"{MODULE_PATH}.CognitiveAgent", side_effect=Exception(error_message))

    start_autonomous_training(tmp_path / "b.json", tmp_path / "s.json")

    mock_app_components["sleep"].assert_not_called()
    logger = mock_app_components["logger"]
    logger.critical.assert_called_once()
    call_args, _ = logger.critical.call_args
    assert error_message in str(call_args[1])


@patch(f"{MODULE_PATH}.start_autonomous_training")
@patch(f"{MODULE_PATH}.setup_logging")
@patch(f"{MODULE_PATH}.argparse.ArgumentParser")
def test_main_function(mock_parser, mock_setup_logging, mock_start_training):
    """Tests that the main() entry point parses args and calls the core function."""
    mock_args = argparse.Namespace(
        goal="cli goal", brain_file=Path("b.json"), state_file=Path("s.json")
    )
    mock_parser.return_value.parse_args.return_value = mock_args

    main()

    mock_setup_logging.assert_called_once()
    mock_parser.return_value.parse_args.assert_called_once()
    mock_start_training.assert_called_once_with(
        Path("b.json"), Path("s.json"), initial_goal="cli goal"
    )
