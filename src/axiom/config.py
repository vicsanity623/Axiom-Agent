from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# --- Core Project Structure (largely unchanged) ---
# These paths point to the source code and should not be changed.
THIS_FILE: Final = Path(__file__).resolve()
PROJECT_ROOT: Final = THIS_FILE.parent.parent.parent

SRC_ROOT: Final = PROJECT_ROOT / "src"
AXIOM_DIR: Final = SRC_ROOT / "axiom"
STATIC_DIR: Final = AXIOM_DIR / "static"
TEMPLATE_DIR: Final = AXIOM_DIR / "templates"

# --- REFACTOR: Data and Model paths are now configurable via environment variables ---
# This allows for flexible deployment, especially with Docker, by allowing all
# mutable data to be stored in a separate volume.

# Use AXIOM_DATA_DIR to specify a custom root for all mutable data.
# It defaults to the project root for a standard development setup.
_data_dir_str = os.getenv("AXIOM_DATA_DIR", str(PROJECT_ROOT))
DATA_ROOT: Final = Path(_data_dir_str)

# Directories for models, brain files, and rendered outputs are now relative
# to the configurable DATA_ROOT.
MODELS_DIR: Final = DATA_ROOT / "models"
BRAIN_DIR: Final = DATA_ROOT / "brain"
RENDERED_DIR: Final = DATA_ROOT / "rendered"

# --- File Paths (derived from the configurable directories) ---
DEFAULT_BRAIN_FILE: Final = BRAIN_DIR / "my_agent_brain.json"
DEFAULT_STATE_FILE: Final = BRAIN_DIR / "my_agent_state.json"
DEFAULT_CACHE_FILE: Final = BRAIN_DIR / "interpreter_cache.json"
MODEL_VERSION_FILE: Final = BRAIN_DIR / "model_version.txt"

# The LLM model name itself can also be configured via an environment variable.
DEFAULT_MODEL_NAME: Final = os.getenv(
    "AXIOM_MODEL_NAME", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
DEFAULT_LLM_PATH: Final = MODELS_DIR / DEFAULT_MODEL_NAME

# --- NEW: Centralized API Key Management ---
# API keys are loaded securely from environment variables.
GEMINI_API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN: Final[str | None] = os.getenv(
    "GITHUB_TOKEN"
)  # For future auto-PR feature
NGROK_AUTHTOKEN: Final[str | None] = os.getenv("NGROK_AUTHTOKEN")
