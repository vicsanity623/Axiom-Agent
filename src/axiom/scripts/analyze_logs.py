"""CLI tool to manually trigger the PerformanceMonitor on Axiom logs."""

import argparse
import logging
from pathlib import Path

from ..logging_config import setup_logging
from ..metacognitive_engine import PerformanceMonitor

logger = logging.getLogger(__name__)


def main():
    """Parse CLI arguments and analyze the specified Axiom log file for optimization targets."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Analyze Axiom logs to detect recurring errors or slow functions.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("axiom.log"),
        help="Path to the log file to analyze (default: axiom.log).",
    )
    args = parser.parse_args()

    if not args.log.exists():
        logger.error("Log file not found at: %s", args.log)
        return

    logger.info("--- [Log Analyzer]: Starting analysis of '%s'... ---", args.log)
    monitor = PerformanceMonitor()
    result = monitor.find_optimization_target(args.log)

    if result:
        logger.info("\n[bold green]=== Optimization Target Identified ===[/bold green]")
        logger.info("[cyan]File:[/cyan] %s", result.file_path)
        logger.info("[cyan]Function:[/cyan] %s", result.target_name)
        logger.info("[cyan]Issue:[/cyan] %s", result.issue_description)
        logger.info("\n[bold cyan]Relevant Logs:[/bold cyan]\n%s", result.relevant_logs)
    else:
        logger.info(
            "\n[bold green]No significant optimization targets found.[/bold green]"
        )


if __name__ == "__main__":
    main()
