from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading

import tornado.web
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from terminado import SingleTermManager, TermSocket

# ── Project Imports ─────────────────────────────────────────────────────
from axiom.cognitive_agent import CognitiveAgent
from axiom.config import (
    DEFAULT_BRAIN_FILE,
    DEFAULT_STATE_FILE,
    PROJECT_ROOT,
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


# ── SSE Log Stream ───────────────────────────────────────────────────────
async def log_generator():
    """Yields log messages from the queue for the SSE stream."""
    while True:
        try:
            line = log_queue.get_nowait()
            yield f"data: {json.dumps({'line': line})}\n\n"
        except queue.Empty:
            await asyncio.sleep(0.1)


# ── Training Thread Target ───────────────────────────────────────────────
def run_training():
    print("Autonomous training started in background thread...")
    try:
        start_autonomous_training(
            brain_file=DEFAULT_BRAIN_FILE,
            state_file=DEFAULT_STATE_FILE,
            initial_goal=None,
            duration_hours=None,
            stop_event=stop_event,
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
    agent = CognitiveAgent(
        brain_file=DEFAULT_BRAIN_FILE,
        state_file=DEFAULT_STATE_FILE,
        inference_mode=False,
    )
    agent.load_state()
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


@app.get("/console", response_class=HTMLResponse)
async def get_console_page():
    path = STATIC_DIR / "studio" / "terminal.html"
    if not path.exists():
        return HTMLResponse("<h1>Error: terminal.html not found!</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


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


@app.post("/api/command")
async def command(cmd: str = Form(...)):
    """Handles interactive commands sent from the IDE input bar as form data."""
    if not agent:
        return HTMLResponse(
            '<div class="log-line text-red-500">Agent not loaded.</div>'
        )
    try:
        response = agent.chat(cmd)
        return HTMLResponse(
            f'<div class="log-line text-gray-400">&gt; {cmd}</div>'
            f'<div class="log-line text-cyan-400">{response}</div>'
        )
    except Exception as e:
        return HTMLResponse(f'<div class="log-line text-red-400">Error: {e}</div>')


@app.get("/api/status")
async def status():
    nodes = edges = deferred = 0
    if agent and agent.graph:
        nodes = len(agent.graph.get_all_nodes())
        edges = len(agent.graph.get_all_edges())
        deferred = len(getattr(agent, "high_priority_tasks", []))

    return HTMLResponse(f"""
        <div id="graph-stats" hx-swap-oob="true" class="text-xs space-y-1">
            <h3 class="text-sm font-semibold text-gray-400 mb-2">Knowledge Graph Stats</h3>
            <div>Nodes: <span class="text-indigo-400 font-mono">{nodes}</span></div>
            <div>Edges: <span class="text-indigo-400 font-mono">{edges}</span></div>
            <div>Deferred: <span class="text-yellow-400 font-mono">{deferred}</span></div>
        </div>
        <script>
            document.body.dispatchEvent(new CustomEvent('updateChart', {{ detail: {{ deferred: {deferred} }} }}));
        </script>
    """)


def start_terminal_server():
    """
    Starts the Tornado server with the Terminado WebSocket backend in a background thread.
    This function is designed to be run in its own thread.
    """
    asyncio.set_event_loop(asyncio.new_event_loop())

    term_manager = SingleTermManager(shell_command=["/bin/zsh"])
    terminal_app = tornado.web.Application(
        [
            (r"/terminal/ws", TermSocket, {"term_manager": term_manager}),
        ]
    )
    terminal_app.listen(8765)
    print("Terminal WebSocket running on ws://127.0.0.1:8765/terminal/ws")

    tornado.ioloop.IOLoop.current().start()


terminal_thread = threading.Thread(target=start_terminal_server, daemon=True)
terminal_thread.start()


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "axiom.scripts.studio_server:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
