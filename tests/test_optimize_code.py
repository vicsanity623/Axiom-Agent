import sys
from pathlib import Path

import pytest

# Ensure the src directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from axiom.metacognitive_engine import (
    BASELINE_STORE_PATH,
    OptimizationTarget,
    PerformanceMonitor,
)


@pytest.fixture(autouse=True)
def clear_baseline_store():
    """Ensures a clean state for each test by removing the baseline file."""
    if BASELINE_STORE_PATH.exists():
        BASELINE_STORE_PATH.unlink()
    yield
    if BASELINE_STORE_PATH.exists():
        BASELINE_STORE_PATH.unlink()


def test_find_optimization_target_identifies_error_pattern(tmp_path: Path):
    """
    Given a log file with a high frequency of identical errors from one function,
    the monitor should identify that function as a target due to error recurrence.
    """
    log_text = "\n".join(
        [
            "2023-01-01T12:00:00 [Symbolic Parser]: ERROR: Failed. (in _parse_single_clause)"
            for _ in range(6)
        ],
    )
    log_file = tmp_path / "error_log.log"
    log_file.write_text(log_text, encoding="utf-8")
    monitor = PerformanceMonitor()

    target = monitor.find_optimization_target(log_file)

    assert isinstance(target, OptimizationTarget)
    assert target.target_name == "_parse_single_clause"
    assert "error recurrence" in target.issue_description.lower()
    assert "anomaly score" in target.issue_description.lower()


def test_no_target_found_for_clean_log(tmp_path: Path):
    """
    Given a log file with no errors or significant performance deviations,
    the monitor should return None, indicating no action is needed.
    """
    log_file = tmp_path / "clean.log"
    log_file.write_text("2023-01-01T12:00:00 [CoreAgent]: INFO: All systems nominal.\n")
    monitor = PerformanceMonitor()

    target = monitor.find_optimization_target(log_file)

    assert target is None


def test_deferred_learning_is_prioritized_as_target(tmp_path: Path):
    """
    Given a log file containing a 'Deferred learning' entry, the monitor
    should prioritize this as a critical target, regardless of other anomalies.
    """
    log_text = "2025-10-24T07:00:00 [Cognitive Agent]: Deferred learning for Fact123"
    log_file = tmp_path / "deferred.log"
    log_file.write_text(log_text, encoding="utf-8")
    monitor = PerformanceMonitor()

    target = monitor.find_optimization_target(log_file)

    assert isinstance(target, OptimizationTarget)
    # FIX: The engine now correctly maps this systemic issue to a real function
    # for the code introspector to analyze.
    assert target.target_name == "CognitiveAgent.__init__"
    assert "deferred learning was triggered" in target.issue_description.lower()
    assert "Log entry: " in target.issue_description
    assert "Fact123" in target.issue_description


def test_baselines_are_saved_even_when_no_target_is_found(tmp_path: Path):
    """
    The monitor should always save its updated statistical baselines at the
    end of a cycle, even if no target meets the threshold for modification.
    """
    log_text = (
        "2023-01-01T12:00:00 [Graph Core]: WARNING: Low confidence edge. (in update_graph)\n"
        "2023-01-01T12:00:01 [Graph Core]: Execution time: 0.1s (in update_graph)"
    )
    log_file = tmp_path / "minor_issues.log"
    log_file.write_text(log_text, encoding="utf-8")
    monitor = PerformanceMonitor()

    assert not BASELINE_STORE_PATH.exists()

    target = monitor.find_optimization_target(log_file)

    assert target is None
    assert BASELINE_STORE_PATH.exists()
    assert "update_graph" in BASELINE_STORE_PATH.read_text(encoding="utf-8")


def test_identifies_inefficient_learning_cycle(tmp_path: Path):
    """
    Given a log file showing a high rate of simple, duplicate facts, and low
    concept discovery, the monitor should identify a systemic learning issue.
    """
    log_entries = [
        "Learned new fact: A --[is]--> B (status=inserted)",
        "Learned new fact: A --[is]--> B (status=inserted)",
        "Learned new fact: A --[is]--> B (status=inserted)",
        "Learned new fact: A --[is]--> B (status=inserted)",
        "Learned new fact: A --[is]--> B (status=inserted)",
        "Learned new fact: B --[is]--> C (status=inserted)",
        "Learned new fact: B --[is]--> C (status=inserted)",
        "Learned new fact: B --[is]--> C (status=inserted)",
        "Learned new fact: C --[is]--> A (status=inserted)",
        "Learned new fact: C --[is]--> A (status=inserted)",
    ]
    log_text = "\n".join(
        f"2023-01-01T12:00:00 [CognitiveAgent]: INFO: {entry}" for entry in log_entries
    )
    log_file = tmp_path / "inefficient_learning.log"
    log_file.write_text(log_text, encoding="utf-8")
    monitor = PerformanceMonitor()

    target = monitor.find_optimization_target(log_file)

    assert isinstance(target, OptimizationTarget)
    # FIX: This systemic issue is now correctly mapped to a real function.
    assert target.target_name == "CognitiveAgent.__init__"
    assert target.file_path == Path("src/axiom/cognitive_agent.py")

    description = target.issue_description.lower()
    assert "high ratio of simple facts" in description
    assert "high ratio of duplicate facts" in description
    assert "low new concept discovery rate" in description
