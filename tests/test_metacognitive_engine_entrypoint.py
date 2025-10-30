from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

# Import the app object from the script under test
from axiom.__main__ import app

# Create a runner that can invoke the CLI commands
runner = CliRunner()


@pytest.fixture
def mock_performance_monitor(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture to mock the PerformanceMonitor and its methods."""
    mock = MagicMock()
    monkeypatch.setattr("axiom.__main__.PerformanceMonitor", mock)
    return mock


@pytest.fixture
def mock_cognitive_agent(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture to mock the CognitiveAgent."""
    mock = MagicMock()
    monkeypatch.setattr("axiom.__main__.CognitiveAgent", mock)
    return mock


@pytest.fixture
def mock_metacognitive_engine(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture to mock the MetacognitiveEngine and its methods."""
    mock = MagicMock()
    monkeypatch.setattr("axiom.__main__.MetacognitiveEngine", mock)
    return mock


# --- Tests for the `analyze-logs` command ---


def test_analyze_logs_target_found(
    mock_performance_monitor: MagicMock, tmp_path: Path
) -> None:
    """Verify `analyze-logs` prints JSON when a target is found."""
    # Setup: Create a dummy log file and a mock optimization target
    log_file = tmp_path / "test.log"
    log_file.touch()

    mock_target = MagicMock()
    mock_target.file_path = Path("src/axiom/test_file.py")
    mock_target.target_name = "test_function"
    mock_target.issue_description = "Test issue"
    mock_target.relevant_logs = "Log line 1"
    mock_performance_monitor.return_value.find_optimization_target.return_value = (
        mock_target
    )

    # Run the command
    result = runner.invoke(app, ["analyze-logs", str(log_file)])

    # Assertions
    assert result.exit_code == 0
    assert "✅ Found an optimization target:" in result.stdout

    # This is not dependent on the exact line number.
    json_start_index = result.stdout.find("{")
    assert json_start_index != -1, "Could not find the start of the JSON output."

    json_string = result.stdout[json_start_index:]
    output_json = json.loads(json_string)

    assert output_json["target_name"] == "test_function"
    assert output_json["file_path"] == "src/axiom/test_file.py"
    mock_performance_monitor.return_value.find_optimization_target.assert_called_once_with(
        log_file
    )


def test_analyze_logs_no_target_found(
    mock_performance_monitor: MagicMock, tmp_path: Path
) -> None:
    """Verify `analyze-logs` prints a success message when no target is found."""
    # Setup: Mock the monitor to return None
    log_file = tmp_path / "test.log"
    log_file.touch()
    mock_performance_monitor.return_value.find_optimization_target.return_value = None

    # Run the command
    result = runner.invoke(app, ["analyze-logs", str(log_file)])

    # Assertions
    assert result.exit_code == 0
    assert (
        "✅ Analysis complete. No high-priority optimization targets found."
        in result.stdout
    )
    mock_performance_monitor.return_value.find_optimization_target.assert_called_once_with(
        log_file
    )


def test_analyze_logs_file_not_exist() -> None:
    """Verify the command fails gracefully if the log file does not exist."""
    # Run the command with a path that is guaranteed not to exist
    result = runner.invoke(app, ["analyze-logs", "non_existent_file.log"])

    # Assertions: Typer's built-in validation should handle this
    assert result.exit_code != 0

    # FIX: Check result.stderr for the error message, not result.stdout.
    assert "does not exist" in result.stderr


# --- Tests for the `cycle-now` command ---


def test_cycle_now_success(
    mock_cognitive_agent: MagicMock, mock_metacognitive_engine: MagicMock
) -> None:
    """Verify `cycle-now` runs a full cycle successfully."""
    # Run the command
    result = runner.invoke(app, ["cycle-now"])

    # Assertions
    assert result.exit_code == 0
    assert "🚀 Initializing agent" in result.stdout
    assert "⚙️ Running introspection cycle..." in result.stdout
    assert "🏁 Cycle complete." in result.stdout

    # Verify that the core components were instantiated and used
    mock_cognitive_agent.assert_called_once()
    mock_metacognitive_engine.assert_called_once()
    mock_metacognitive_engine.return_value.run_introspection_cycle.assert_called_once()


def test_cycle_now_with_force_and_existing_file(
    mock_cognitive_agent: MagicMock,
    mock_metacognitive_engine: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--force` deletes an existing suggestion file before running."""
    # Setup: Create a dummy suggestion file and change to its directory
    suggestion_file = tmp_path / "code_suggestion.json"
    suggestion_file.write_text("existing suggestion")
    monkeypatch.chdir(tmp_path)

    assert suggestion_file.exists()

    # Run the command with the --force flag
    result = runner.invoke(app, ["cycle-now", "--force"])

    # Assertions
    assert result.exit_code == 0
    assert "🗑️ --force specified. Removing existing suggestion file" in result.stdout
    # Verify the file was deleted before the cycle ran
    assert not suggestion_file.exists()
    mock_metacognitive_engine.return_value.run_introspection_cycle.assert_called_once()


def test_cycle_now_without_force_keeps_file(
    mock_cognitive_agent: MagicMock,
    mock_metacognitive_engine: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that without `--force`, an existing file is NOT deleted."""
    # Setup
    suggestion_file = tmp_path / "code_suggestion.json"
    suggestion_file.write_text("existing suggestion")
    monkeypatch.chdir(tmp_path)

    assert suggestion_file.exists()

    # Run the command without the flag
    result = runner.invoke(app, ["cycle-now"])

    # Assertions
    assert result.exit_code == 0
    assert "Removing existing suggestion file" not in result.stdout
    assert suggestion_file.exists()  # File should still be there
    mock_metacognitive_engine.return_value.run_introspection_cycle.assert_called_once()


def test_cycle_now_handles_critical_error(
    mock_cognitive_agent: MagicMock, mock_metacognitive_engine: MagicMock
) -> None:
    """Verify the command aborts on an exception during initialization."""
    # Setup: Make one of the core components fail on instantiation
    error_message = "Test initialization failure"
    mock_cognitive_agent.side_effect = Exception(error_message)

    # Run the command
    result = runner.invoke(app, ["cycle-now"])

    # Assertions
    assert result.exit_code != 0
    assert (
        f"❌ CRITICAL ERROR during metacognitive cycle: {error_message}"
        in result.stdout
    )
    # The engine's run method should not have been called
    mock_metacognitive_engine.return_value.run_introspection_cycle.assert_not_called()


# --- Tests for placeholder commands ---


def test_introspect_command() -> None:
    """Verify the `introspect` command prints its placeholder message and exits."""
    result = runner.invoke(app, ["introspect"])

    # typer.Exit() results in a clean exit code of 0
    assert result.exit_code == 0
    assert "Feature not yet implemented" in result.stdout


def test_self_modify_command() -> None:
    """Verify the `self-modify` command prints its placeholder message and exits."""
    result = runner.invoke(app, ["self-modify"])

    assert result.exit_code == 0
    assert "Feature not yet implemented" in result.stdout
