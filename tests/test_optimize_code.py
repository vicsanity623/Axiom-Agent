import os
import sys
from pathlib import Path

import pytest

# Ensure src is in sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from axiom.metacognitive_engine import (
    BASELINE_STORE_PATH,
    BaselineStats,
    OptimizationTarget,
    PerformanceMonitor,
)


# Automatically clear baseline store before each test
@pytest.fixture(autouse=True)
def clear_baseline_store():
    if BASELINE_STORE_PATH.exists():
        os.remove(BASELINE_STORE_PATH)


def make_baseline_stats(
    mean_time: float = 0.2,
    var_time: float = 0.0001,  # very small variance for deterministic testing
    samples: int = 3,
    error_count_ewma: float = 0.0,
) -> BaselineStats:
    """Helper to create a dummy BaselineStats object with stable attributes for testing."""
    return BaselineStats(
        mean_time=mean_time,
        var_time=var_time,
        samples=samples,
        error_count_ewma=error_count_ewma,
    )


def test_find_optimization_target_identifies_error_pattern(tmp_path: Path):
    """Test that the monitor identifies a recurring function failure."""
    log_text = "\n".join(
        [
            "[Symbolic Parser]: ERROR: Failed. No clauses could be parsed. (in _parse_single_clause)"
            for _ in range(6)
        ],
    )

    log_file = tmp_path / "test_log.log"
    log_file.write_text(log_text, encoding="utf-8")

    monitor = PerformanceMonitor()
    target = monitor.find_optimization_target(log_file)

    assert isinstance(target, OptimizationTarget)
    assert target.target_name == "_parse_single_clause"
    assert "error recurrence" in target.issue_description.lower()


def test_no_target_found(tmp_path: Path):
    """If log is clean, no optimization target should be returned."""
    log_file = tmp_path / "clean.log"
    log_file.write_text("[CoreAgent]: INFO: All systems nominal.\n")

    monitor = PerformanceMonitor()
    target = monitor.find_optimization_target(log_file)

    assert target is None
