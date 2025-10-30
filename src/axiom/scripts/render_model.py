from __future__ import annotations

import json
import logging
import re
import time
import zipfile
from datetime import UTC, datetime

from ..config import (
    BRAIN_DIR,
    DEFAULT_BRAIN_FILE,
    DEFAULT_CACHE_FILE,
    MODEL_VERSION_FILE,
    RENDERED_DIR,
)

logger = logging.getLogger(__name__)


def get_next_version() -> str:
    """Reads the current version, increments it, and saves it back."""
    major = 0
    minor = 0

    BRAIN_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_VERSION_FILE.exists():
        try:
            version_str = MODEL_VERSION_FILE.read_text(encoding="utf-8")
            match = re.search(r"(\d+)\.(\d+)", version_str)
            if match:
                major, minor = map(int, match.groups())
        except Exception:
            logger.warning(
                "   - Could not parse version file. Starting from 0.0.", exc_info=True
            )

    minor += 1
    if minor >= 10:
        major += 1
        minor = 0

    next_version = f"{major}.{minor}"

    MODEL_VERSION_FILE.write_text(next_version, encoding="utf-8")

    return next_version


def clear_old_models(current_version_str: str) -> None:
    """Deletes all .axm files in the rendered folder EXCEPT the current one."""
    current_filename = RENDERED_DIR / f"Axiom_{current_version_str}.axm"
    logger.info("   - Cleaning up old models in '%s'...", RENDERED_DIR)

    if not RENDERED_DIR.exists():
        return

    for file_path in RENDERED_DIR.glob("Axiom_*.axm"):
        if file_path.resolve() != current_filename.resolve():
            try:
                file_path.unlink()
                logger.info("     - Deleted old model: %s", file_path.name)
            except OSError as e:
                logger.warning(
                    "     - Could not delete %s. Error: %s", file_path.name, e
                )


def render_axiom_model() -> None:
    """
    Package the agent's current brain and cache into a single, versioned .axm model.
    This hardened version includes validation and locking to prevent corrupt models.
    """
    brain_file = DEFAULT_BRAIN_FILE
    cache_file = DEFAULT_CACHE_FILE
    lock_file = BRAIN_DIR / f"{brain_file.name}.lock"

    version_str = get_next_version()
    logger.info("--- Starting Axiom Mind Renderer [Version: %s] ---", version_str)

    # --- Step 1: Wait for File Lock ---
    max_wait_seconds = 10
    wait_interval = 1
    waited_time = 0
    while lock_file.exists():
        if waited_time >= max_wait_seconds:
            logger.critical(
                "Lock file '%s' persisted for over %d seconds. An 'axiom-train' process may be stuck. Aborting render.",
                lock_file,
                max_wait_seconds,
            )
            return
        logger.info("   - Waiting for lock file '%s' to be released...", lock_file.name)
        time.sleep(wait_interval)
        waited_time += wait_interval

    # --- Step 2: Validate Brain File Content ---
    if not brain_file.exists():
        logger.critical("Source file '%s' not found.", brain_file)
        logger.critical(
            "   Please ensure the agent has been run at least once to generate its brain."
        )
        return

    try:
        brain_content = brain_file.read_text(encoding="utf-8")
        if not brain_content.strip():
            logger.critical("Brain file '%s' is empty. Aborting render.", brain_file)
            return

        # Try to parse the JSON to ensure it's valid
        brain_data = json.loads(brain_content)
        logger.info("✅ Brain file is valid and contains data.")

    except json.JSONDecodeError:
        logger.critical(
            "Brain file '%s' contains invalid JSON. Aborting render.", brain_file
        )
        return
    except Exception as e:
        logger.critical("Failed to read or validate brain file. Error: %s", e)
        return

    # --- Step 3: Proceed with Packaging ---
    RENDERED_DIR.mkdir(exist_ok=True)
    output_filename = RENDERED_DIR / f"Axiom_{version_str}.axm"
    version_data = {
        "model_format": "AxiomMind",
        "version": version_str,
        "render_date_utc": datetime.now(UTC).isoformat(),
    }

    try:
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            # Use the validated content we already loaded
            zf.writestr("brain.json", json.dumps(brain_data))
            logger.info("   - Compressing validated brain data...")

            if cache_file.exists():
                zf.write(cache_file, arcname="cache.json")
                logger.info("   - Compressing %s...", cache_file)
            else:
                empty_cache: dict[str, list] = {"interpretations": [], "synthesis": []}
                zf.writestr("cache.json", json.dumps(empty_cache))
                logger.info("   - Cache file not found. Packaging an empty cache.")

            zf.writestr("version.json", json.dumps(version_data, indent=2))
            logger.info("   - Compressing version.json...")

        logger.info("\n✅ Successfully rendered model: %s", output_filename)

        clear_old_models(version_str)

    except Exception as e:
        logger.critical("Failed to create the .axm package. Error: %s", e)


def main() -> None:
    render_axiom_model()


if __name__ == "__main__":
    main()
