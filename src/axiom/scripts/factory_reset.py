from __future__ import annotations

import logging
from pathlib import Path

from ..config import DEFAULT_BRAIN_FILE, MASTER_BRAIN_FILE
from ..logging_config import setup_logging

logger = logging.getLogger(__name__)

# Define all target files for a full factory reset.
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
    """Entry point for the axiom-factory-reset command."""
    setup_logging()

    project_root = find_project_root(Path.cwd())
    if not project_root:
        logger.error(
            "Could not find project root. Please run from within the project directory."
        )
        return

    logger.warning("[bold red]!!! FACTORY RESET !!![/bold red]")
    logger.warning(
        "[bold yellow]This will permanently delete ALL of the agent's learned knowledge, state, and logs.[/bold yellow]"
    )
    logger.warning("This action cannot be undone.")

    try:
        confirm = input("To confirm this destructive action, type 'yes': ")
    except (KeyboardInterrupt, EOFError):
        logger.info("\nFactory reset cancelled by user.")
        return

    if confirm.lower() != "yes":
        logger.info("Confirmation failed. Factory reset cancelled.")
        return

    logger.info("\n--- [Axiom Factory Reset]: Wiping all data... ---")
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

    # Optional but recommended: Restore the master brain
    master_brain_path = project_root / MASTER_BRAIN_FILE
    user_brain_path = project_root / DEFAULT_BRAIN_FILE
    if master_brain_path.exists() and not user_brain_path.exists():
        try:
            import shutil

            shutil.copy(master_brain_path, user_brain_path)
            logger.info("   - Restored master brain to user brain file.")
        except Exception as e:
            logger.error("   - FAILED to restore master brain. Reason: %s", e)

    logger.info(
        "\n[bold green]✅ Factory Reset complete. Deleted %d file(s).[/bold green]",
        deleted_count,
    )


if __name__ == "__main__":
    main()
