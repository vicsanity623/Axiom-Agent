from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import traceback

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

from ..cognitive_agent import CognitiveAgent
from ..config import DEFAULT_BRAIN_FILE, DEFAULT_STATE_FILE, STATIC_DIR, TEMPLATE_DIR
from ..knowledge_harvester import KnowledgeHarvester
from ..logging_config import setup_logging
from ..metacognitive_engine import MetacognitiveEngine
from .cycle_manager import CycleManager

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

axiom_agent: CognitiveAgent | None = None
agent_interaction_lock = threading.Lock()
agent_status: str = "uninitialized"


def load_agent() -> None:
    """Initialize the global Axiom Agent and start its autonomous cycles."""
    global axiom_agent, agent_status
    loading_lock = threading.Lock()
    with loading_lock:
        if axiom_agent is None and agent_status != "loading":
            agent_status = "loading"
            try:
                logger.info("--- Starting Axiom Agent Initialization... ---")

                axiom_agent = CognitiveAgent(
                    brain_file=DEFAULT_BRAIN_FILE,
                    state_file=DEFAULT_STATE_FILE,
                )

                harvester = KnowledgeHarvester(
                    agent=axiom_agent,
                    lock=agent_interaction_lock,
                )
                scheduler = BackgroundScheduler(daemon=True)

                gemini_api_key = os.environ.get("GEMINI_API_KEY")
                metacognitive_engine = MetacognitiveEngine(
                    agent=axiom_agent,
                    gemini_api_key=gemini_api_key,
                )
                manager = CycleManager(scheduler, harvester, metacognitive_engine)

                manager.start()

                scheduler.start()

                agent_status = "ready"
                logger.info("--- Axiom Agent is Ready! ---")
            except Exception as exc:
                agent_status = f"error: {exc}"
                logger.critical(
                    "!!! CRITICAL ERROR INITIALIZING AGENT: %s !!!",
                    exc,
                    exc_info=True,
                )
                traceback.print_exc()


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


@app.route("/status")
def status() -> str | Response:
    """Provide the agent's loading status and trigger initialization."""
    global agent_status
    if agent_status == "uninitialized":
        threading.Thread(target=load_agent).start()
        agent_status = "loading"
    return jsonify({"status": agent_status})


@app.route("/chat", methods=["POST"])
def chat() -> tuple[Response, int] | Response:
    """Handle an incoming user message and return the agent's response."""
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
            logger.debug("  [Lock]: User chat acquired lock.")
            agent_response = axiom_agent.chat(user_message)
        logger.debug("  [Lock]: User chat released lock.")
        return jsonify({"response": agent_response})
    except Exception as e:
        logger.error("!!! ERROR DURING CHAT PROCESSING: %s !!!", e, exc_info=True)
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


def run() -> None:
    """Parse command-line arguments and start the Flask web server."""
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
            logger.warning(
                "[ngrok Warning]: NGROK_AUTHTOKEN environment variable not set. Using anonymous tunnel.",
            )

        public_url = ngrok.connect(7500)
        logger.info(" * ngrok tunnel is active at: %s", public_url)

    app.run(host="0.0.0.0", port=7500, debug=False)


def main() -> None:
    setup_logging()
    run()


if __name__ == "__main__":
    main()
