from __future__ import annotations

import logging
from pathlib import Path

from ..logging_config import setup_logging

logger = logging.getLogger(__name__)

# --- FIX: Define only the files for a SOFT reset ---
# This list intentionally OMITS my_agent_brain.json
FILES_TO_DELETE: list[str] = [
    "code_suggestion.json",
    "axiom.log",
    "data/research_cache.json",
    "src/axiom/brain/my_agent_state.json",
    "src/axiom/brain/interpreter_cache.json",
]


def find_project_root(start_path: Path) -> Path | None:
    """Traverse up from start_path to find the directory containing 'pyproject.toml'."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def main() -> None:
    """
    Entry point for the axiom-reset command. Performs a SOFT reset, preserving
    the agent's long-term knowledge (the brain file).
    """
    setup_logging()

    project_root = find_project_root(Path.cwd())
    if not project_root:
        logger.error(
            "Could not find project root. Please run from within the project directory."
        )
        return

    logger.info("[bold yellow]Performing a SOFT reset.[/bold yellow]")
    logger.info(
        "This will reset the agent's current state, goals, and caches, but will PRESERVE its learned knowledge."
    )
    logger.info(
        "To perform a full factory reset, use the 'axiom-factory-reset' command."
    )

    # A soft reset is less destructive, so we can make the confirmation simpler.
    try:
        confirm = input("Are you sure you want to continue? (yes/no): ")
    except (KeyboardInterrupt, EOFError):
        logger.info("\nReset cancelled by user.")
        return

    if confirm.lower() != "yes":
        logger.info("Reset cancelled.")
        return

    logger.info("\n--- [Axiom Soft Reset]: Resetting agent state... ---")
    deleted_count = 0
    for file_str in FILES_TO_DELETE:
        file_path = project_root / file_str
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info("   - Deleted: %s", file_path)
                deleted_count += 1
            except OSError as e:
                logger.error("   - FAILED to delete: %s. Reason: %s", file_path, e)
        else:
            logger.debug("   - Not found, skipping: %s", file_path)

    logger.info(
        "\n[bold green]✅ Soft Reset complete. Deleted %d file(s).[/bold green]",
        deleted_count,
    )


if __name__ == "__main__":
    main()
