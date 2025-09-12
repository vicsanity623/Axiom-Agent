# render_model.py

import os
import sys
import json
import zipfile
from datetime import datetime

def render_axiom_model(version: str):
    """
    Finds the current agent state files and packages them into a versioned .axm model file.
    """
    print(f"--- Starting Axiom Mind Renderer for version {version} ---")

    # 1. Define the required source files
    brain_file = "my_agent_brain.json"
    cache_file = "interpreter_cache.json"
    
    source_files = [brain_file, cache_file]

    # 2. Check if all source files exist
    for f in source_files:
        if not os.path.exists(f):
            print(f"❌ CRITICAL ERROR: Source file '{f}' not found.")
            print("Please ensure the agent has been run at least once to generate its brain.")
            return

    print("✅ Found all required source files.")

    # 3. Create the version metadata file
    version_data = {
        "model_format": "AxiomMind",
        "version": version,
        "render_date_utc": datetime.utcnow().isoformat()
    }
    version_filename = "version.json"
    with open(version_filename, 'w') as f:
        json.dump(version_data, f, indent=2)
    
    print(f"✅ Created version metadata file for v{version}.")

    # 4. Create the final .axm package (which is a zip archive)
    output_filename = f"axiom_v{version}.axm"
    try:
        with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add the brain with a generic name inside the archive
            zf.write(brain_file, arcname='brain.json')
            print(f"   - Compressing {brain_file}...")
            
            # Add the cache with a generic name
            zf.write(cache_file, arcname='cache.json')
            print(f"   - Compressing {cache_file}...")

            # Add the version file
            zf.write(version_filename, arcname='version.json')
            print(f"   - Compressing {version_filename}...")
        
        print(f"\n✅ Successfully rendered model: {output_filename}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to create the .axm package. Error: {e}")
    finally:
        # 5. Clean up the temporary version file
        if os.path.exists(version_filename):
            os.remove(version_filename)
            print("🧹 Cleaned up temporary files.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 render_model.py <version_number>")
        print("Example: python3 render_model.py 0.0.1")
    else:
        model_version = sys.argv[1]
        render_axiom_model(model_version)