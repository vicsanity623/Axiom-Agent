from __future__ import annotations

import glob
import json

# setup/app_model.py
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Final

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from axiom.cognitive_agent import CognitiveAgent

axiom_agent: CognitiveAgent | None = None

THIS_FILE: Final = Path(__file__).absolute()
STATIC_DIR: Final = THIS_FILE.parent / "static"
TEMPLATE_DIR = THIS_FILE.parent / "templates"

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


def find_latest_model(directory: str = "rendered") -> str | None:
    """Find and return the path to the most recent .axm model file.

    Scans the specified directory for files matching the 'axiom_model_*.axm'
    pattern, sorts them lexicographically (which works for timestamps),
    and returns the full path to the latest one.

    Args:
        directory: The directory to search for model files.

    Returns:
        The file path to the latest model as a string, or None if no
        model files are found.
    """
    print(
        f"--- [Server]: Searching for the latest .axm model file in '{directory}'... ---",
    )
    search_pattern = os.path.join(directory, "axiom_model_*.axm")
    model_files = glob.glob(search_pattern)
    if not model_files:
        print(
            f"!!! [Server]: CRITICAL ERROR: No .axm model files found in '{directory}'.",
        )
        return None
    latest_file = sorted(model_files, reverse=True)[0]
    print(f"--- [Server]: Found latest model: '{os.path.basename(latest_file)}' ---")
    return latest_file


def load_axiom_model(axm_filepath: str) -> tuple[Any, Any] | None:
    """Load and unpack an Axiom Mind (.axm) model file.

    Reads the specified .axm file (which is a zip archive) and extracts
    the `brain.json` and `cache.json` data. Prints version information
    from the model's metadata upon successful loading.

    Args:
        axm_filepath: The absolute or relative path to the .axm model file.

    Returns:
        A tuple containing the brain_data and cache_data dictionaries,
        or None if the file does not exist or fails to parse.
    """
    if not os.path.exists(axm_filepath):
        return None
    try:
        with zipfile.ZipFile(axm_filepath, "r") as zf:
            brain_data = json.loads(zf.read("brain.json"))
            cache_data = json.loads(zf.read("cache.json"))
            version_data = json.loads(zf.read("version.json"))
            print("--- [Server]: Axiom Mind Data Successfully Read ---")
            print(f"  - Version: {version_data.get('version')}")
            print(f"  - Render Date (UTC): {version_data.get('render_date_utc')}")
            return brain_data, cache_data
    except Exception as e:
        print(
            f"!!! [Server]: CRITICAL ERROR: Failed to load or parse .axm model. Error: {e}",
        )
        return None


@app.route("/")
def index() -> str:
    """Serve the main single-page application HTML."""
    return render_template("index.html")


@app.route("/manifest.json")
def manifest() -> Response:
    """Serve the PWA manifest file for web app installation."""
    return send_from_directory(STATIC_DIR, "manifest.json")


@app.route("/sw.js")
def service_worker() -> Response:
    """Serve the service worker script for PWA offline capabilities."""
    return send_from_directory(STATIC_DIR, "sw.js")


@app.route("/chat", methods=["POST"])
def chat() -> tuple[Response, int] | Response:
    """Handle incoming user messages and return the agent's response.

    This is the main API endpoint for conversation. It expects a JSON
    payload with a 'message' key. It passes this message to the loaded
    Axiom Agent's chat method and returns the agent's reply in a
    JSON object with a 'response' key.

    Returns:
        A JSON response with the agent's reply, or a JSON error
        object with an appropriate HTTP status code on failure.
    """
    if not axiom_agent:
        return jsonify({"error": "Agent is not available or is still loading."}), 503

    if request.json is None:
        return jsonify({"error": "request.json is None"}), 503

    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    try:
        agent_response = axiom_agent.chat(user_message)
        return jsonify({"response": agent_response})
    except Exception as e:
        print(f"!!! [Server]: ERROR DURING CHAT PROCESSING: {e} !!!")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


@app.route("/status")
def status() -> Response:
    """Provide the current loading status of the agent.

    This endpoint allows the front-end to poll the server to determine
    if the Axiom Agent has been successfully initialized from the model
    file.

    Returns:
        A JSON response with a 'status' key, which will be 'ready'
        or 'loading_model'.
    """
    return jsonify({"status": "ready" if axiom_agent else "loading_model"})


def run() -> int:
    """Run webserver."""
    global axiom_agent

    latest_model_path = find_latest_model()
    if not latest_model_path:
        print(
            "--- [Server]: Shutting down. Please run a trainer to generate a model first.",
        )
        return 1

    model_data = load_axiom_model(latest_model_path)

    if not model_data:
        return 1

    brain, cache = model_data
    print("--- [Server]: Initializing agent in INFERENCE-ONLY mode... ---")
    axiom_agent = CognitiveAgent(
        load_from_file=False,
        brain_data=brain,
        cache_data=cache,
        inference_mode=True,
    )
    print("--- [Server]: Agent is ready. Starting web server... ---")
    app.run(host="0.0.0.0", port=7501, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(run())
