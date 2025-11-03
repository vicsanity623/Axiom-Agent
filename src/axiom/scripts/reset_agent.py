# In src/axiom/scripts/reset_agent.py

from __future__ import annotations

import logging
from pathlib import Path

from ..logging_config import setup_logging

logger = logging.getLogger(__name__)

# Define all target files relative to a potential project root.
# This makes the script runnable from any directory within the project.
FILES_TO_DELETE: list[str] = [
    "code_suggestion.json",
    "axiom.log",
    "data/research_cache.json",
    "src/axiom/brain/my_agent_state.json",
    "src/axiom/brain/my_agent_brain.json",
    "src/axiom/brain/interpreter_cache.json",
    "src/axiom/brain/model_version.txt",
    "src/axiom/rendered/Axiom_0.3.axm",
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
    """Entry point for the axiom-reset command."""
    setup_logging()

    project_root = find_project_root(Path.cwd())
    if not project_root:
        logger.error(
            "Could not find project root (containing 'pyproject.toml'). "
            "Please run this command from within the Axiom-Agent project directory."
        )
        return

    logger.warning(
        "[bold yellow]This command will permanently delete the agent's brain, state, logs, and caches.[/bold yellow]"
    )
    logger.warning("This action cannot be undone.")

    try:
        confirm = input("Are you sure you want to continue? (yes/no): ")
    except (KeyboardInterrupt, EOFError):
        logger.info("\nReset cancelled by user.")
        return

    if confirm.lower() != "yes":
        logger.info("Reset cancelled.")
        return

    logger.info("\n--- [Axiom Reset]: Wiping the slate clean... ---")
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
        "\n[bold green]✅ Reset complete. Deleted %d file(s).[/bold green]",
        deleted_count,
    )


if __name__ == "__main__":
    main()
