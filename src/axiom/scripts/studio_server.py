from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from pathlib import Path

import tornado.web
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from terminado import SingleTermManager, TermSocket

# ── Project Imports ─────────────────────────────────────────────────────
from axiom.cognitive_agent import CognitiveAgent
from axiom.config import (
    DEFAULT_BRAIN_FILE,
    DEFAULT_STATE_FILE,
    STATIC_DIR,
)
from axiom.scripts.autonomous_trainer import start_autonomous_training

# ── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(title="Axiom Studio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
agent: CognitiveAgent | None = None
training_thread: threading.Thread | None = None
stop_event = threading.Event()
log_queue: queue.Queue[str] = queue.Queue()


# ── Logging to Queue (captures ALL axiom logs) ──────────────────────────
class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue[str]):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put(self.format(record))
        except Exception:
            pass


def setup_queue_logging():
    logger = logging.getLogger("axiom")
    if not any(isinstance(h, QueueLogHandler) for h in logger.handlers):
        handler = QueueLogHandler(log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_training():
    print("Autonomous training started in background thread...")

    def update_agent(new_agent):
        global agent
        agent = new_agent
        print("--- [Studio Server]: Global agent updated from training thread. ---")

    try:
        start_autonomous_training(
            brain_file=DEFAULT_BRAIN_FILE,
            state_file=DEFAULT_STATE_FILE,
            initial_goal=None,
            duration_hours=None,
            stop_event=stop_event,
            agent_callback=update_agent,
        )
    except Exception as e:
        logging.error(f"Training crashed: {e}")
    finally:
        print("Training thread ended.")


# ── FastAPI Endpoints ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    setup_queue_logging()
    global agent
    print("Initializing interactive CognitiveAgent...")
    # Initialize a read-only agent first, will be replaced if training starts
    agent = CognitiveAgent(
        brain_file=DEFAULT_BRAIN_FILE,
        state_file=DEFAULT_STATE_FILE,
        inference_mode=False,  # Enable learning mode for Axiom Studio
    )
    try:
        agent.load_state()
    except Exception:
        print("Warning: Could not load initial agent state.")

    # Start terminal server in background
    global terminal_thread
    terminal_thread = threading.Thread(target=start_terminal_server, daemon=True)
    terminal_thread.start()

    print("Axiom Studio ready → http://localhost:8000/studio")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/studio")


@app.get("/studio")
async def studio():
    path = STATIC_DIR / "studio" / "index.html"
    if not path.exists():
        return HTMLResponse(
            "Missing src/axiom/static/studio/index.html", status_code=500
        )
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/chat")
async def chat_page():
    path = STATIC_DIR / "studio" / "chat.html"
    if not path.exists():
        return HTMLResponse(
            "Missing src/axiom/static/studio/chat.html", status_code=500
        )
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/welcome")
async def welcome_page():
    path = STATIC_DIR / "studio" / "welcome.html"
    if not path.exists():
        return HTMLResponse(
            "Missing src/axiom/static/studio/welcome.html", status_code=500
        )
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/agent_manager")
async def agent_manager_page():
    path = STATIC_DIR / "studio" / "agent_manager.html"
    if not path.exists():
        return HTMLResponse("Missing agent_manager.html", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/clone")
async def clone_page():
    path = STATIC_DIR / "studio" / "clone.html"
    if not path.exists():
        return HTMLResponse("Missing clone.html", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/recent_workspaces")
async def get_recent_workspaces():
    """Returns a list of recent workspace paths from the local config."""
    config_path = Path.home() / ".axiom" / "studio_config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return data.get("recent_workspaces", [])
        except Exception:
            pass
    return []


class WorkspaceItem(BaseModel):
    path: str


@app.delete("/api/recent_workspaces")
async def remove_recent_workspace(item: WorkspaceItem):
    """Removes a workspace path from the recent list."""
    config_path = Path.home() / ".axiom" / "studio_config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            recents = data.get("recent_workspaces", [])
            if item.path in recents:
                recents.remove(item.path)
                data["recent_workspaces"] = recents
                config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return {"status": "ok"}
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)
    return {"status": "ok"}


class AgentSettings(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    system_prompt: str = ""


@app.get("/api/agent_settings")
async def get_agent_settings():
    config_path = Path.home() / ".axiom" / "agent_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return AgentSettings().dict()


@app.post("/api/agent_settings")
async def save_agent_settings(settings: AgentSettings):
    config_dir = Path.home() / ".axiom"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "agent_config.json"
    config_path.write_text(json.dumps(settings.dict(), indent=2), encoding="utf-8")
    return {"status": "ok"}


class CloneRequest(BaseModel):
    repo_url: str
    destination: str


@app.post("/api/clone")
async def clone_repo(req: CloneRequest):
    import subprocess

    dest_path = Path(req.destination)
    if not dest_path.parent.exists():
        return JSONResponse(
            {"detail": "Parent directory does not exist"}, status_code=400
        )

    if dest_path.exists() and any(dest_path.iterdir()):
        return JSONResponse({"detail": "Destination is not empty"}, status_code=400)

    try:
        # Run git clone
        process = subprocess.run(
            ["git", "clone", req.repo_url, str(dest_path)],
            capture_output=True,
            text=True,
        )

        if process.returncode == 0:
            return {
                "message": f"Successfully cloned to {dest_path}",
                "path": str(dest_path),
            }
        return JSONResponse({"detail": process.stderr}, status_code=500)

    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/console", response_class=HTMLResponse)
async def get_console_page():
    path = STATIC_DIR / "studio" / "terminal.html"
    if not path.exists():
        return HTMLResponse("<h1>Error: terminal.html not found!</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


async def log_generator():
    """Yields new log lines from the queue as Server-Sent Events."""
    while True:
        try:
            # Non-blocking get
            line = log_queue.get_nowait()
            yield f"data: {line}\n\n"
        except queue.Empty:
            await asyncio.sleep(0.1)


@app.get("/api/log_stream")
async def log_stream():
    return StreamingResponse(log_generator(), media_type="text/event-stream")


@app.post("/api/start-training")
async def start_training():
    global training_thread
    if training_thread and training_thread.is_alive():
        return HTMLResponse(
            '<div class="log-line text-yellow-400">Training already running.</div>'
        )

    stop_event.clear()
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    return HTMLResponse(
        '<div class="log-line text-green-400">Autonomous training started.</div>'
    )


@app.post("/api/stop-training")
async def stop_training():
    if not training_thread or not training_thread.is_alive():
        return HTMLResponse(
            '<div class="log-line text-gray-500">No training running.</div>'
        )

    stop_event.set()
    return HTMLResponse(
        '<div class="log-line text-red-400">Stop signal sent. Waiting for graceful shutdown...</div>'
    )

    stop_event.set()
    return HTMLResponse(
        '<div class="log-line text-red-400">Stop signal sent. Waiting for graceful shutdown...</div>'
    )


@app.post("/api/reset_chat")
async def reset_chat():
    """Clears the agent's conversation history."""
    if agent:
        # Clear symbolic agent history (ConversationManager is stateless per request, so no history to clear)
        # if hasattr(agent, "conversation_manager"):
        #     pass

        # Clear coding agent history (LLM)
        if hasattr(agent, "coding_agent"):
            agent.coding_agent.clear_history()

        return HTMLResponse(
            '<div class="log-line text-gray-500">--- Chat History Cleared ---</div>'
        )
    return HTMLResponse('<div class="log-line text-red-500">Agent not ready.</div>')


@app.post("/api/command")
async def command(
    cmd: str = Form(...),
    active_file_name: str = Form(None),
    active_file_content: str = Form(None),
    coding_mode: str = Form(None),
    axiom_mode: str = Form(None),
):
    """Handles interactive commands sent from the IDE input bar as form data."""
    if not agent:
        return HTMLResponse(
            '<div class="message agent"><div class="message-content">Agent not loaded.</div></div>'
        )
    try:
        # Construct context object
        context = {}
        if active_file_name and active_file_content:
            context["active_file"] = {
                "name": active_file_name,
                "content": active_file_content,
            }

        response = ""

        # Routing Logic
        if axiom_mode == "true":
            # Axiom Mode: Use CognitiveAgent (Symbolic + LLM)
            response = agent.chat(cmd)
        elif coding_mode == "true" or cmd.startswith("/code"):
            # Code Mode: Use CodingAgent (Direct LLM)
            if hasattr(agent, "coding_agent"):
                response = agent.coding_agent.process_request(cmd, context=context)
            else:
                response = "Coding Agent not initialized."
        else:
            # Default: Use CodingAgent for standard chat (Gemma 3 experience)
            if hasattr(agent, "coding_agent"):
                response = agent.coding_agent.process_request(cmd, context=context)
            else:
                response = agent.chat(cmd)

        # Return HTML bubble
        return HTMLResponse(
            f'<div class="message agent"><div class="message-content">{response}</div></div>'
        )
    except Exception as e:
        return HTMLResponse(
            f'<div class="message agent"><div class="message-content">Error: {e}</div></div>'
        )


@app.get("/api/status")
async def status():
    nodes = edges = deferred = 0

    # Read stats directly from disk to ensure sync with external training process
    try:
        if DEFAULT_BRAIN_FILE.exists():
            brain_data = json.loads(DEFAULT_BRAIN_FILE.read_text(encoding="utf-8"))
            nodes = len(brain_data.get("nodes", {}))
            edges = len(brain_data.get("links", {}))

        if DEFAULT_STATE_FILE.exists():
            state_data = json.loads(DEFAULT_STATE_FILE.read_text(encoding="utf-8"))
            # Count high priority tasks
            deferred += len(state_data.get("high_priority_tasks", []))
            # Also count learning goals as deferred work?
            # The user said "deferred count shows zero when in fact there are several tasks deferred"
            # Let's include learning_goals in the count as they are pending work.
            deferred += len(state_data.get("learning_goals", []))

    except Exception as e:
        print(f"Error reading stats from disk: {e}")
        # Fallback to in-memory agent if disk read fails
        if agent and agent.graph:
            try:
                nodes = len(agent.graph.get_all_nodes())
                edges = len(agent.graph.get_all_edges())
                deferred = len(getattr(agent, "high_priority_tasks", []))
            except Exception:
                pass

    from fastapi.responses import JSONResponse

    return JSONResponse({"nodes": nodes, "edges": edges, "deferred": deferred})


@app.get("/api/curriculum", response_class=HTMLResponse)
async def get_curriculum():
    """Returns the active curriculum/goal state as an HTML fragment."""
    try:
        if not DEFAULT_STATE_FILE.exists():
            return HTMLResponse(
                '<div class="text-gray-500 text-sm">Waiting for agent state...</div>',
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        state_data = json.loads(DEFAULT_STATE_FILE.read_text(encoding="utf-8"))
        # If the state file is empty or lacks goals, treat as not ready
        if not state_data or not state_data.get("goals"):
            return HTMLResponse(
                '<div class="text-gray-500 text-sm">Waiting for agent state...</div>',
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        goals = state_data.get("goals", {})

        active_goal = None
        active_stage = None

        # Find active goal/stage
        for goal in goals.values():
            if goal.get("status") == "in_progress":
                # Check if it has active stages
                if goal.get("stages"):
                    for stage in goal["stages"]:
                        if stage.get("status") == "in_progress":
                            active_goal = goal
                            active_stage = stage
                            break
                else:
                    active_goal = goal

                if active_goal:
                    break

        if not active_goal:
            return """
            <div class="flex flex-col items-center justify-center h-full text-gray-500 space-y-2">
                <i class="fas fa-brain text-2xl opacity-50"></i>
                <span class="text-sm">No active curriculum. Agent is idle.</span>
            </div>
            """

        # Render Active Goal
        html = '<div class="flex flex-col gap-4 h-full">'

        # Main Goal Header
        html += f"""
        <div class="bg-[#27272a] rounded-lg p-3 border border-[#3f3f46]">
            <div class="text-[10px] uppercase tracking-wider text-indigo-400 font-bold mb-1">Current Objective</div>
            <div class="text-sm text-gray-200 font-medium leading-snug">
                {active_goal.get("description", "Unknown Goal")}
            </div>
        </div>
        """

        # Active Stage (if any)
        if active_stage:
            html += f"""
            <div class="pl-3 border-l-2 border-indigo-500/30">
                <div class="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Current Stage</div>
                <div class="text-sm text-gray-300">
                    {active_stage.get("description", "").split(":")[-1].strip()}
                </div>
            </div>
            """

        # Upcoming Tasks (Sub-goals)
        # If we have a stage, use its sub-goals. Otherwise use main goal's.
        current_context = active_stage if active_stage else active_goal
        sub_goals = current_context.get("sub_goals", [])

        # Filter sub_goals to only show tasks that are still in the learning_goals queue
        learning_goals_set = set()
        for lg in state_data.get("learning_goals", []):
            topic = (
                lg.replace("INVESTIGATE:", "").strip()
                if isinstance(lg, str)
                else lg.get("topic", "")
            )
            learning_goals_set.add(topic.lower())

        # Only show sub_goals that are still pending (in learning_goals)
        pending_sub_goals = []
        for sg in sub_goals:
            topic = (
                sg.replace("INVESTIGATE:", "").strip()
                if isinstance(sg, str)
                else sg.get("topic", "")
            )
            if topic.lower() in learning_goals_set:
                pending_sub_goals.append(sg)

        if pending_sub_goals:
            html += '<div class="flex-1 overflow-hidden flex flex-col">'
            html += '<div class="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Immediate Tasks</div>'
            html += (
                '<div class="flex-1 overflow-y-auto pr-2 space-y-2 custom-scrollbar">'
            )

            # Show top 10 pending tasks
            for i, task in enumerate(pending_sub_goals[:10]):
                task_text = (
                    task if isinstance(task, str) else task.get("topic", str(task))
                )
                task_text = task_text.replace("INVESTIGATE:", "").strip()

                # Highlight first task as active
                status_icon = (
                    '<i class="fas fa-circle-notch fa-spin text-indigo-400 text-[10px]"></i>'
                    if i == 0
                    else '<i class="far fa-circle text-gray-600 text-[10px]"></i>'
                )
                text_color = "text-indigo-100" if i == 0 else "text-gray-400"

                html += f"""
                <div class="flex items-start gap-2 text-xs {text_color}">
                    <div class="mt-1">{status_icon}</div>
                    <div class="leading-relaxed">{task_text}</div>
                </div>
                """

            if len(pending_sub_goals) > 10:
                html += f'<div class="text-[10px] text-gray-600 pl-5">... and {len(pending_sub_goals) - 10} more</div>'

            html += "</div></div>"

        html += "</div>"
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    except Exception as e:
        return f'<div class="text-red-500 text-xs">Error loading curriculum: {e}</div>'


def start_terminal_server():
    """
    Starts the Tornado server with the Terminado WebSocket backend in a background thread.
    This function is designed to be run in its own thread.
    """
    asyncio.set_event_loop(asyncio.new_event_loop())

    term_manager = SingleTermManager(shell_command=["/bin/zsh"])

    class AllowOriginTermSocket(TermSocket):
        def check_origin(self, origin):
            return True

    terminal_app = tornado.web.Application(
        [
            (r"/terminal/ws", AllowOriginTermSocket, {"term_manager": term_manager}),
        ]
    )
    terminal_app.listen(8765)
    print("Terminal WebSocket running on ws://127.0.0.1:8765/terminal/ws")

    tornado.ioloop.IOLoop.current().start()


# terminal_thread = threading.Thread(target=start_terminal_server, daemon=True)
# terminal_thread.start()


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "axiom.scripts.studio_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


@app.get("/api/system_stats")
async def system_stats():
    """Returns real-time system resource usage (CPU, RAM, GPU)."""
    try:
        import psutil

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # RAM Usage
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)
        ram_percent = memory.percent

        # GPU Usage (optional, may not be available)
        gpu_percent = 0
        gpu_memory_used_gb = 0
        gpu_memory_total_gb = 0

        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_percent = gpu.load * 100
                gpu_memory_used_gb = gpu.memoryUsed / 1024
                gpu_memory_total_gb = gpu.memoryTotal / 1024
        except Exception:
            # GPU monitoring not available
            pass

        return JSONResponse(
            {
                "cpu_percent": round(cpu_percent, 1),
                "ram_used_gb": round(ram_used_gb, 2),
                "ram_total_gb": round(ram_total_gb, 2),
                "ram_percent": round(ram_percent, 1),
                "gpu_percent": round(gpu_percent, 1),
                "gpu_memory_used_gb": round(gpu_memory_used_gb, 2),
                "gpu_memory_total_gb": round(gpu_memory_total_gb, 2),
            }
        )
    except Exception as e:
        return JSONResponse(
            {
                "cpu_percent": 0,
                "ram_used_gb": 0,
                "ram_total_gb": 0,
                "ram_percent": 0,
                "gpu_percent": 0,
                "gpu_memory_used_gb": 0,
                "gpu_memory_total_gb": 0,
                "error": str(e),
            }
        )


@app.post("/api/cancel_chat")
async def cancel_chat():
    """Cancels the currently running chat request."""
    try:
        if agent and hasattr(agent, "coding_agent"):
            agent.coding_agent.cancel_generation()
            return HTMLResponse(
                '<div class="message agent"><div class="message-content">[Response cancelled by user]</div></div>'
            )
        return HTMLResponse(
            '<div class="message agent"><div class="message-content">[Nothing to cancel]</div></div>'
        )
    except Exception as e:
        return HTMLResponse(
            f'<div class="message agent"><div class="message-content">[Cancel failed: {str(e)}]</div></div>'
        )
