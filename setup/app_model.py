from __future__ import annotations

import glob
import json

# setup/app_model.py
import os
import sys
import zipfile

from flask import Flask, jsonify, render_template, request, send_from_directory

from axiom.cognitive_agent import CognitiveAgent

axiom_agent = None

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
TEMPLATE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "templates"),
)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


def find_latest_model(directory="rendered"):
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


def load_axiom_model(axm_filepath):
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
def index():
    return render_template("index.html")


@app.route("/manifest.json")
def manifest():
    return send_from_directory(STATIC_DIR, "manifest.json")


@app.route("/sw.js")
def service_worker():
    return send_from_directory(STATIC_DIR, "sw.js")


@app.route("/chat", methods=["POST"])
def chat():
    if not axiom_agent:
        return jsonify({"error": "Agent is not available or is still loading."}), 503
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
def status():
    return jsonify({"status": "ready" if axiom_agent else "loading_model"})


if __name__ == "__main__":
    latest_model_path = find_latest_model()
    if not latest_model_path:
        print(
            "--- [Server]: Shutting down. Please run a trainer to generate a model first.",
        )
        sys.exit(1)

    model_data = load_axiom_model(latest_model_path)

    if model_data:
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
    else:
        sys.exit(1)
