"""CLI tool to analyze Axiom logs for performance and optimization targets."""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from axiom.metacognitive_engine import PerformanceMonitor

logger = logging.getLogger("AxiomCLI")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    """Parse CLI arguments and analyze the specified Axiom log file for optimization targets."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Axiom logs to detect recurring errors or slow functions.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the log file to analyze.",
    )
    args = parser.parse_args()

    monitor = PerformanceMonitor()
    result = monitor.find_optimization_target(args.log)

    if result:
        print("\n=== Optimization Target Identified ===")
        print(f"File: {result.file_path}")
        print(f"Function: {result.target_name}")
        print(f"Issue: {result.issue_description}")
        print("\nRelevant Logs:\n" + result.relevant_logs)
    else:
        print("\nNo optimization targets found.")


if __name__ == "__main__":
    main()
