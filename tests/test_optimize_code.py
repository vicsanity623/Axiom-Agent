import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from axiom.metacognitive_engine import (
    BASELINE_STORE_PATH,
    CodeIntrospector,
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


@pytest.fixture
def mock_introspector() -> MagicMock:
    """Provides a mock CodeIntrospector that can find dummy file paths."""
    introspector = MagicMock(spec=CodeIntrospector)
    introspector.get_source_file.return_value = Path("src/axiom/dummy_file.py")
    return introspector


def test_no_target_found_for_clean_log(tmp_path: Path, mock_introspector: MagicMock):
    """
    Given a log file with no errors or significant performance deviations,
    the monitor should return None, indicating no action is needed.
    """
    log_file = tmp_path / "clean.log"
    log_file.write_text("2023-01-01T12:00:00 [CoreAgent]: INFO: All systems nominal.\n")
    monitor = PerformanceMonitor()

    target = monitor.find_optimization_target(log_file, mock_introspector)

    assert target is None


def test_baselines_are_saved_even_when_no_target_is_found(
    tmp_path: Path, mock_introspector: MagicMock, monkeypatch: Any
):
    """
    The monitor should always save its updated statistical baselines at the
    end of a cycle, even if no target meets the threshold for modification.
    """
    log_text = (
        "[12:00:00] WARNING Low confidence edge. (in update_graph)\n"
        "[12:00:01] INFO Execution time: 0.1s (in update_graph)"
    )
    log_file = tmp_path / "minor_issues.log"
    log_file.write_text(log_text, encoding="utf-8")

    baseline_path = tmp_path / "test_baseline.json"
    monitor = PerformanceMonitor(baseline_store=baseline_path)

    assert not baseline_path.exists()

    target = monitor.find_optimization_target(log_file, mock_introspector)

    assert target is None
    assert baseline_path.exists()
    assert "update_graph" in baseline_path.read_text(encoding="utf-8")
