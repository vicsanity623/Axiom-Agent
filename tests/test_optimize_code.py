import sys
from pathlib import Path

import pytest

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
