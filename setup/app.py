from __future__ import annotations

import argparse
import os
import threading
import time
import traceback
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from pyngrok import ngrok

from axiom.cognitive_agent import CognitiveAgent
from axiom.knowledge_harvester import KnowledgeHarvester

app = Flask(__name__)

axiom_agent: CognitiveAgent | None = None
agent_interaction_lock = threading.Lock()
agent_status: str = "uninitialized"
DEFAULT_BRAIN_FILE = Path("my_agent_brain.json")
DEFAULT_STATE_FILE = Path("my_agent_state.json")


def load_agent() -> None:
    """Initialize the global Axiom Agent and start its autonomous cycles.

    This function is the core initialization routine for the training
    server. It performs a thread-safe, one-time setup of the
    CognitiveAgent, loading its brain from disk.

    It then creates and schedules the KnowledgeHarvester's two main
    autonomous learning cycles (Study and Discovery) to run in the
    background, updating a global status variable to reflect the
    agent's state ('loading', 'ready', or 'error').
    """
    global axiom_agent, agent_status
    loading_lock = threading.Lock()
    with loading_lock:
        if axiom_agent is None and agent_status != "loading":
            agent_status = "loading"
            try:
                print("--- Starting Axiom Agent Initialization... ---")

                axiom_agent = CognitiveAgent(
                    brain_file=DEFAULT_BRAIN_FILE,
                    state_file=DEFAULT_STATE_FILE,
                )

                harvester = KnowledgeHarvester(
                    agent=axiom_agent,
                    lock=agent_interaction_lock,
                )
                scheduler = BackgroundScheduler(daemon=True)

                scheduler.add_job(
                    harvester.study_cycle,
                    "interval",
                    minutes=6,
                )
                print("--- Study Cycle is scheduled to run every 6 minutes. ---")

                scheduler.add_job(
                    harvester.discover_cycle,
                    "interval",
                    hours=35,
                )
                print("--- Discovery Cycle is scheduled to run every 35 minutes. ---")

                scheduler.start()

                agent_status = "ready"
                print("--- Axiom Agent is Ready! ---")
            except Exception as exc:
                agent_status = f"error: {exc}"
                print(f"!!! CRITICAL ERROR INITIALIZING AGENT: {exc} !!!")

                traceback.print_exc()


@app.route("/")
def index() -> str:
    """Serve the main single-page application HTML."""
    return render_template("index.html")


@app.route("/manifest.json")
def manifest() -> Response:
    """Serve the PWA manifest file for web app installation."""
    return send_from_directory("static", "manifest.json")


@app.route("/sw.js")
def service_worker() -> Response:
    """Serve the service worker script for PWA offline capabilities."""
    return send_from_directory("static", "sw.js")


@app.route("/status")
def status() -> str | Response:
    """Provide the agent's loading status and trigger initialization.

    This endpoint is polled by the front-end. On the very first call,
    it triggers the `load_agent` function in a background thread to
    begin the resource-intensive initialization process.

    Subsequent calls will return the current status ('loading', 'ready',
    or 'error') without re-triggering the load.

    Returns:
        A JSON response with a 'status' key indicating the agent's state.
    """
    global agent_status
    if agent_status == "uninitialized":
        threading.Thread(target=load_agent).start()
        agent_status = "loading"
    return jsonify({"status": agent_status})


@app.route("/chat", methods=["POST"])
def chat() -> tuple[Response, int] | Response:
    """Handle an incoming user message and return the agent's response.

    This is the main API endpoint for conversation. It waits until the
    agent is fully initialized, then accepts a JSON payload with a
    'message' key.

    The user's message is passed to the agent's chat method within a
    thread-safe lock. The agent's reply is returned in a JSON object
    with a 'response' key.

    Returns:
        A JSON response with the agent's reply, or a JSON error
        object with an appropriate HTTP status code on failure.
    """
    start_time = time.time()
    while agent_status != "ready":
        if agent_status.startswith("error"):
            return jsonify({"error": f"Agent failed to load: {agent_status}"}), 500
        if time.time() - start_time > 300:
            return jsonify({"error": "Agent is taking too long to initialize."}), 503
        time.sleep(1)

    if request.json is None:
        return jsonify({"error": "Request json attribute is None"}), 503

    if axiom_agent is None:
        return jsonify({"error": "axiom_agent is None"}), 503

    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        with agent_interaction_lock:
            print("  [Lock]: User chat acquired lock.")
            agent_response = axiom_agent.chat(user_message)
        print("  [Lock]: User chat released lock.")
        return jsonify({"response": agent_response})
    except Exception as e:
        print(f"!!! ERROR DURING CHAT PROCESSING: {e} !!!")

        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


def run() -> None:
    """Parse command-line arguments and start the Flask web server.

    This function sets up the main application entry point. It handles
    the optional '--ngrok' flag to expose the local server to the
    internet via a public URL.
    """
    parser = argparse.ArgumentParser(description="Run the Axiom Agent Training App.")
    parser.add_argument(
        "--ngrok",
        action="store_true",
        help="Expose the server to the internet using ngrok.",
    )
    args = parser.parse_args()

    if args.ngrok:
        authtoken = os.environ.get("NGROK_AUTHTOKEN")
        if authtoken:
            ngrok.set_auth_token(authtoken)
        else:
            print(
                "[ngrok Warning]: NGROK_AUTHTOKEN environment variable not set. Using anonymous tunnel.",
            )

        public_url = ngrok.connect(7500)
        print(f" * ngrok tunnel is active at: {public_url}")

    app.run(host="0.0.0.0", port=7500, debug=False)


if __name__ == "__main__":
    run()
