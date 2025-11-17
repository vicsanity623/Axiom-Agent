from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from ..cognitive_agent import CognitiveAgent
from ..config import RENDERED_DIR, STATIC_DIR, TEMPLATE_DIR

logger = logging.getLogger(__name__)
axiom_agent: CognitiveAgent | None = None


app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))


def find_latest_model(directory: Path = RENDERED_DIR) -> Path | None:
    """Find and return the path to the most recent .axm model file.

    Scans the specified directory for files matching the 'Axiom_*.axm'
    pattern, sorts them by modification time (most recent first), and
    returns the path to the latest one.

    Args:
        directory: The directory to search for model files.

    Returns:
        The file path to the latest model as a Path object, or None if no
        model files are found.
    """
    logger.info("Searching for the latest .axm model file in '%s'...", directory)
    try:
        model_files = list(directory.glob("Axiom_*.axm"))

        if not model_files:
            logger.critical("No .axm model files found in '%s'.", directory)
            return None

        latest_file = sorted(
            model_files,
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[0]

        logger.info("Found latest model: '%s'", latest_file.name)
        return latest_file
    except Exception as e:
        logger.critical("Error while searching for latest model: %s", e, exc_info=True)
        return None


def load_axiom_model(axm_filepath: Path) -> tuple[dict, dict] | None:
    """Load, verify, and unpack an Axiom Mind (.axm) model file.

    Reads the specified .axm zip archive, performs security checks,
    verifies the checksum of the brain file, and extracts the brain
    and cache data.

    Args:
        axm_filepath: The path to the .axm model file.

    Returns:
        A tuple containing the brain_data and cache_data dictionaries,
        or None if the file is invalid, corrupt, or fails verification.
    """
    if not axm_filepath.exists():
        logger.error(
            "Attempted to load a model file that does not exist: %s",
            axm_filepath,
        )
        return None

    try:
        with zipfile.ZipFile(axm_filepath, "r") as zf:
            for member in zf.namelist():
                if ".." in member or member.startswith("/"):
                    raise ValueError(
                        f"Security risk: Invalid path in zip file: {member}"
                    )

            version_data = json.loads(zf.read("version.json"))
            brain_bytes = zf.read("brain.json")
            cache_data = json.loads(zf.read("cache.json"))

            schema_version = version_data.get("schema_version", 1)
            expected_checksum = version_data.get("checksum")

            if schema_version != 1:
                raise ValueError(f"Unsupported schema version: {schema_version}")

            if expected_checksum:
                calculated_checksum = hashlib.sha256(brain_bytes).hexdigest()
                if calculated_checksum != expected_checksum:
                    raise ValueError(
                        f"Checksum mismatch! Brain may be corrupt. Expected {expected_checksum}, got {calculated_checksum}",
                    )
                logger.info("Brain checksum verified successfully.")

            brain_data = json.loads(brain_bytes)

            logger.info("Axiom Mind Data Successfully Read")
            logger.info("  - Version: %s", version_data.get("version"))
            logger.info(
                "  - Render Date (UTC): %s",
                version_data.get("render_date_utc"),
            )
            return brain_data, cache_data
    except (zipfile.BadZipFile, KeyError) as e:
        logger.critical(
            "Failed to read .axm archive '%s'. It may be corrupt or missing required files (brain.json, cache.json, version.json). Error: %s",
            axm_filepath.name,
            e,
        )
        return None
    except json.JSONDecodeError as e:
        logger.critical(
            "Failed to parse JSON within '%s'. The model data is corrupt. Error: %s",
            axm_filepath.name,
            e,
        )
        return None
    except ValueError as e:
        logger.critical(
            "Model verification failed for '%s'. Error: %s", axm_filepath.name, e
        )
        return None
    except Exception as e:
        logger.critical(
            "An unexpected error occurred while loading .axm model '%s': %s",
            axm_filepath.name,
            e,
            exc_info=True,
        )
        return None


@app.route("/")
def index() -> str:
    """Serve the main single-page application HTML."""
    return render_template("index.html")


@app.route("/manifest.json")
def manifest() -> Response:
    """Serve the PWA manifest file for web app installation."""
    return send_from_directory(str(STATIC_DIR), "manifest.json")


@app.route("/sw.js")
def service_worker() -> Response:
    """Serve the service worker script for PWA offline capabilities."""
    return send_from_directory(str(STATIC_DIR), "sw.js")


@app.route("/chat", methods=["POST"])
def chat() -> tuple[Response, int] | Response:
    """Handle incoming user messages and return the agent's response."""
    if not axiom_agent:
        return jsonify({"error": "Agent is not available or is still loading."}), 503

    if not request.json or "message" not in request.json:
        return jsonify(
            {"error": "Invalid request: missing JSON or 'message' key."},
        ), 400

    user_message = request.json["message"]
    try:
        agent_response = axiom_agent.chat(user_message)
        return jsonify({"response": agent_response})
    except Exception as e:
        logger.exception("An internal error occurred during chat processing")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


@app.route("/status")
def status() -> Response:
    """Provide the current loading status of the agent."""
    return jsonify({"status": "ready" if axiom_agent else "loading_model"})


def main() -> int:
    """Load the latest model, initialize the agent, and run the webserver."""
    parser = argparse.ArgumentParser(description="Run the Axiom Agent web UI server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="The host IP address to bind the server to. Default is 0.0.0.0 (accessible on the network).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7501,
        help="The port to run the server on. Default is 7501.",
    )
    args = parser.parse_args()

    global axiom_agent

    latest_model_path = find_latest_model()
    if not latest_model_path:
        logger.critical(
            "Shutting down. Please run a trainer to generate a model first.",
        )
        return 1

    model_data = load_axiom_model(latest_model_path)
    if not model_data:
        logger.critical("Failed to initialize agent from model. Shutting down.")
        return 1

    brain, cache = model_data
    logger.info("Initializing agent in INFERENCE-ONLY mode...")
    axiom_agent = CognitiveAgent(
        load_from_file=False,
        brain_data=brain,
        cache_data=cache,
        inference_mode=True,
    )
    logger.info(
        "Agent is ready. Starting web server on http://%s:%d", args.host, args.port
    )
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
