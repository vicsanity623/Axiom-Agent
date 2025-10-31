from __future__ import annotations

import os
from pathlib import Path
from typing import Final

THIS_FILE: Final = Path(__file__).resolve()
PROJECT_ROOT: Final = THIS_FILE.parent.parent.parent

SRC_ROOT: Final = PROJECT_ROOT / "src"
AXIOM_DIR: Final = SRC_ROOT / "axiom"
STATIC_DIR: Final = AXIOM_DIR / "static"
TEMPLATE_DIR: Final = AXIOM_DIR / "templates"


_data_dir_str = os.getenv("AXIOM_DATA_DIR", str(PROJECT_ROOT))
DATA_ROOT: Final = Path(_data_dir_str)

MODELS_DIR: Final = DATA_ROOT / "models"
BRAIN_DIR: Final = AXIOM_DIR / "brain"
RENDERED_DIR: Final = AXIOM_DIR / "rendered"

DEFAULT_BRAIN_FILE: Final = BRAIN_DIR / "my_agent_brain.json"
DEFAULT_STATE_FILE: Final = BRAIN_DIR / "my_agent_state.json"
DEFAULT_CACHE_FILE: Final = BRAIN_DIR / "interpreter_cache.json"
MODEL_VERSION_FILE: Final = BRAIN_DIR / "model_version.txt"

DEFAULT_MODEL_NAME: Final = os.getenv(
    "AXIOM_MODEL_NAME", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
DEFAULT_LLM_PATH: Final = MODELS_DIR / DEFAULT_MODEL_NAME

GEMINI_API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN: Final[str | None] = os.getenv("GITHUB_TOKEN")
NGROK_AUTHTOKEN: Final[str | None] = os.getenv("NGROK_AUTHTOKEN")
