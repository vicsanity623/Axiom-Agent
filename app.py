# app.py

from flask import Flask, render_template, request, jsonify, send_from_directory
from cognitive_agent import CognitiveAgent
import threading
import time
from apscheduler.schedulers.background import BackgroundScheduler
from knowledge_harvester import KnowledgeHarvester

app = Flask(__name__)

# --- Global Agent Initialization ---
axiom_agent = None
# --- NEW: A global lock to protect the agent from simultaneous access ---
agent_interaction_lock = threading.Lock()
# --- END NEW ---
agent_status = "uninitialized"

def load_agent():
    """
    Function to load the CognitiveAgent and start the background harvester.
    """
    global axiom_agent, agent_status
    # Use a separate lock for the initial loading process to avoid deadlocks
    loading_lock = threading.Lock()
    with loading_lock:
        if axiom_agent is None and agent_status != "loading":
            agent_status = "loading"
            try:
                print("--- Starting Axiom Agent Initialization... ---")
                brain_file = "my_agent_brain.json"
                state_file = "my_agent_state.json"
                
                axiom_agent = CognitiveAgent(brain_file=brain_file, state_file=state_file)
                
                # --- Pass the new interaction lock to the harvester ---
                harvester = KnowledgeHarvester(agent=axiom_agent, lock=agent_interaction_lock)
                scheduler = BackgroundScheduler(daemon=True)
                
                # For testing, you can change 'hours=1' to 'minutes=X' to see it run quickly.
                scheduler.add_job(harvester.harvest_and_learn, 'interval', minutes=12)
                
                scheduler.start()
                print("--- Knowledge Harvester is scheduled to run every 12 minutes (for testing). ---")

                agent_status = "ready"
                print("--- Axiom Agent is Ready! ---")
            except Exception as e:
                agent_status = f"error: {e}"
                print(f"!!! CRITICAL ERROR INITIALIZING AGENT: {e} !!!")
                import traceback
                traceback.print_exc()

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main chat page."""
    return render_template('index.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def service_worker():
    return send_from_directory('static', 'sw.js')

@app.route('/status')
def status():
    """An endpoint for the front-end to poll the agent's loading status."""
    global agent_status
    if agent_status == "uninitialized":
        threading.Thread(target=load_agent).start()
        agent_status = "loading"
    return jsonify({"status": agent_status})

@app.route('/chat', methods=['POST'])
def chat():
    """
    The main endpoint for handling chat messages from the user.
    """
    start_time = time.time()
    while agent_status != "ready":
        if agent_status.startswith("error"):
            return jsonify({"error": f"Agent failed to load: {agent_status}"}), 500
        if time.time() - start_time > 300: # 5 minute timeout for loading
            return jsonify({"error": "Agent is taking too long to initialize."}), 503
        time.sleep(1)

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # --- NEW: Acquire the lock before using the agent ---
        with agent_interaction_lock:
            print("  [Lock]: User chat acquired lock.")
            agent_response = axiom_agent.chat(user_message)
        print("  [Lock]: User chat released lock.")
        # --- END NEW ---
        return jsonify({"response": agent_response})
    except Exception as e:
        print(f"!!! ERROR DURING CHAT PROCESSING: {e} !!!")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    import argparse
    from pyngrok import ngrok
    import os

    parser = argparse.ArgumentParser(description="Run the Axiom Agent Training App.")
    parser.add_argument('--ngrok', action='store_true', help="Expose the server to the internet using ngrok.")
    args = parser.parse_args()

    # Configure and start ngrok if the flag is provided
    if args.ngrok:
        authtoken = os.environ.get("NGROK_AUTHTOKEN")
        if authtoken:
            ngrok.set_auth_token(authtoken)
        else:
            print("[ngrok Warning]: NGROK_AUTHTOKEN environment variable not set. Using anonymous tunnel.")
        
        public_url = ngrok.connect(7500)
        print(f" * ngrok tunnel is active at: {public_url}")

    app.run(host='0.0.0.0', port=7500, debug=False)