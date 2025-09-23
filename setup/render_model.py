from __future__ import annotations

import json

# render_model.py
import os
import zipfile
from datetime import datetime


def render_axiom_model():
    """
    Finds the current agent state files and packages them into a uniquely named .axm model file
    based on the current date and time.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"--- Starting Axiom Mind Renderer [Model ID: {timestamp}] ---")

    brain_file = "brain/my_agent_brain.json"
    cache_file = "brain/interpreter_cache.json"

    source_files = [brain_file, cache_file]

    for f in source_files:
        if not os.path.exists(f):
            print(f"❌ CRITICAL ERROR: Source file '{f}' not found.")
            print(
                "Please ensure the agent has been run at least once to generate its brain.",
            )
            return

    print("✅ Found all required source files.")

    version_data = {
        "model_format": "AxiomMind",
        "version": timestamp,
        "render_date_utc": datetime.utcnow().isoformat(),
    }
    version_filename = "version.json"
    with open(version_filename, "w") as f:
        json.dump(version_data, f, indent=2)

    print(f"✅ Created version metadata file for model {timestamp}.")

    output_filename = f"rendered/axiom_model_{timestamp}.axm"
    try:
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(brain_file, arcname="brain.json")
            print(f"   - Compressing {brain_file}...")

            zf.write(cache_file, arcname="cache.json")
            print(f"   - Compressing {cache_file}...")

            zf.write(version_filename, arcname="version.json")
            print(f"   - Compressing {version_filename}...")

        print(f"\n✅ Successfully rendered model: {output_filename}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to create the .axm package. Error: {e}")
    finally:
        if os.path.exists(version_filename):
            os.remove(version_filename)
            print("🧹 Cleaned up temporary files.")


if __name__ == "__main__":
    render_axiom_model()
