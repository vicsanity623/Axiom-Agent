from __future__ import annotations

import json

# render_model.py
import zipfile
from datetime import datetime
from pathlib import Path


def render_axiom_model() -> None:
    """Package the agent's current brain and cache into a versioned model.

    This script creates a distributable, read-only snapshot of the agent's
    current state. It finds the `my_agent_brain.json` file, optionally
    finds the `interpreter_cache.json` file, and bundles them into a
    single, timestamped `.axm` archive in the `rendered/` directory.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"--- Starting Axiom Mind Renderer [Model ID: {timestamp}] ---")

    brain_folder = Path("brain")
    brain_file = brain_folder / "my_agent_brain.json"
    cache_file = brain_folder / "interpreter_cache.json"

    if not brain_file.exists():
        print(f"❌ CRITICAL ERROR: Source file '{brain_file}' not found.")
        print(
            "Please ensure the agent has been run at least once to generate its brain.",
        )
        return

    print("✅ Found required source files.")

    version_data = {
        "model_format": "AxiomMind",
        "version": timestamp,
        "render_date_utc": datetime.utcnow().isoformat(),
    }
    # version_file = Path("version.json")
    # with version_file.open("w", encoding="utf-8") as fp:
    #     json.dump(version_data, fp, indent=2)

    # print(f"✅ Created version metadata file for model {timestamp}.")

    rendered_folder = Path("rendered")
    output_filename = rendered_folder / f"axiom_model_{timestamp}.axm"
    output_filename.parent.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(brain_file, arcname="brain.json")
            print(f"   - Compressing {brain_file}...")

            if cache_file.exists():
                zf.write(cache_file, arcname="cache.json")
                print(f"   - Compressing {cache_file}...")
            else:
                empty_cache: dict[str, list[object]] = {
                    "interpretations": [],
                    "synthesis": [],
                }
                zf.writestr("cache.json", json.dumps(empty_cache))
                print("   - Cache file not found. Packaging an empty cache.")

            # zf.write(version_file, arcname="version.json")
            zf.writestr("version.json", json.dumps(version_data, indent=2))
            print("   - Compressing version.json...")

        print(f"\n✅ Successfully rendered model: {output_filename}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to create the .axm package. Error: {e}")
    # finally:
    #     if version_file.exists():
    #         version_file.unlink()
    #         print("🧹 Cleaned up temporary files.")


if __name__ == "__main__":
    render_axiom_model()
