# chat_app.py (original read only chat)

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import zipfile

from flask import Flask, jsonify, render_template, request

from axiom.cognitive_agent import CognitiveAgent

# --- Global variable for our loaded agent ---
axiom_agent = None


def load_axiom_model(axm_filepath):
    """
    Loads a .axm model file by unzipping it in memory and returning its contents.
    """
    if not os.path.exists(axm_filepath):
        print(f"❌ CRITICAL ERROR: Model file not found at '{axm_filepath}'")
        return None
    try:
        with zipfile.ZipFile(axm_filepath, "r") as zf:
            brain_data = json.loads(zf.read("brain.json"))
            cache_data = json.loads(zf.read("cache.json"))
            version_data = json.loads(zf.read("version.json"))
            print("--- Axiom Mind Data Successfully Read ---")
            print(f"  - Version: {version_data.get('version')}")
            print(f"  - Render Date (UTC): {version_data.get('render_date_utc')}")
            return brain_data, cache_data
    except Exception as e:
        print(
            f"❌ CRITICAL ERROR: Failed to load or parse the .axm model file. Error: {e}",
        )
        return None


# --- Flask Application Setup ---
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not axiom_agent:
        return jsonify({"error": "Agent is not available."}), 503
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    try:
        agent_response = axiom_agent.chat(user_message)
        return jsonify({"response": agent_response})
    except Exception as e:
        print(f"!!! ERROR DURING CHAT PROCESSING: {e} !!!")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


@app.route("/status")
def status():
    return jsonify({"status": "ready" if axiom_agent else "loading_model"})


if __name__ == "__main__":
    from pyngrok import ngrok

    parser = argparse.ArgumentParser(description="Run the Axiom Agent Chat App.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the .axm model file to load.",
    )
    parser.add_argument(
        "--ngrok",
        action="store_true",
        help="Expose the server to the internet using ngrok.",
    )
    args = parser.parse_args()

    model_data = load_axiom_model(args.model_path)

    if model_data:
        brain, cache = model_data
        # Initialize the agent in "inference mode" with the loaded data
        axiom_agent = CognitiveAgent(
            load_from_file=False,
            brain_data=brain,
            cache_data=cache,
            inference_mode=True,
        )

        # Configure and start ngrok if the flag is provided
        if args.ngrok:
            authtoken = os.environ.get("NGROK_AUTHTOKEN")
            if authtoken:
                ngrok.set_auth_token(authtoken)
            else:
                print(
                    "[ngrok Warning]: NGROK_AUTHTOKEN environment variable not set. Using anonymous tunnel.",
                )

            public_url = ngrok.connect(7501)  # Connect to the correct port for this app
            print(f" * ngrok tunnel is active at: {public_url}")

        app.run(host="0.0.0.0", port=7501, debug=False)
    else:
        sys.exit(1)
