from __future__ import annotations

import json
import re
import zipfile
from datetime import UTC, datetime

from ..config import (
    BRAIN_DIR,
    DEFAULT_BRAIN_FILE,
    DEFAULT_CACHE_FILE,
    MODEL_VERSION_FILE,
    RENDERED_DIR,
)


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
            print("   - Warning: Could not parse version file. Starting from 0.0.")

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
    print(f"   - Cleaning up old models in '{RENDERED_DIR}'...")

    if not RENDERED_DIR.exists():
        return

    for file_path in RENDERED_DIR.glob("Axiom_*.axm"):
        if file_path.resolve() != current_filename.resolve():
            try:
                file_path.unlink()
                print(f"     - Deleted old model: {file_path.name}")
            except OSError as e:
                print(f"     - Warning: Could not delete {file_path.name}. Error: {e}")


def render_axiom_model() -> None:
    """
    Package the agent's current brain and cache into a single, versioned .axm model.
    This script ensures only the latest rendered model exists in the 'rendered/' directory.
    """
    brain_file = DEFAULT_BRAIN_FILE
    cache_file = DEFAULT_CACHE_FILE

    if not brain_file.exists():
        print(f"❌ CRITICAL ERROR: Source file '{brain_file}' not found.")
        print(
            "   Please ensure the agent has been run at least once to generate its brain.",
        )
        return

    version_str = get_next_version()
    print(f"--- Starting Axiom Mind Renderer [Version: {version_str}] ---")

    RENDERED_DIR.mkdir(exist_ok=True)
    output_filename = RENDERED_DIR / f"Axiom_{version_str}.axm"

    print("✅ Found required source files.")

    version_data = {
        "model_format": "AxiomMind",
        "version": version_str,
        "render_date_utc": datetime.now(UTC).isoformat(),
    }

    try:
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(brain_file, arcname="brain.json")
            print(f"   - Compressing {brain_file}...")

            if cache_file.exists():
                zf.write(cache_file, arcname="cache.json")
                print(f"   - Compressing {cache_file}...")
            else:
                empty_cache: dict[str, list] = {"interpretations": [], "synthesis": []}
                zf.writestr("cache.json", json.dumps(empty_cache))
                print("   - Cache file not found. Packaging an empty cache.")

            zf.writestr("version.json", json.dumps(version_data, indent=2))
            print("   - Compressing version.json...")

        print(f"\n✅ Successfully rendered model: {output_filename}")

        clear_old_models(version_str)

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to create the .axm package. Error: {e}")


def main() -> None:
    render_axiom_model()


if __name__ == "__main__":
    main()
