# In src/axiom/config.py

from __future__ import annotations

from pathlib import Path
from typing import Final

# All other modules should import their paths from here.

THIS_FILE: Final = Path(__file__).resolve()
PROJECT_ROOT: Final = THIS_FILE.parent.parent.parent

SRC_ROOT: Final = PROJECT_ROOT / "src"
MODELS_DIR: Final = PROJECT_ROOT / "models"
AXIOM_DIR: Final = SRC_ROOT / "axiom"
BRAIN_DIR: Final = AXIOM_DIR / "brain"
RENDERED_DIR: Final = AXIOM_DIR / "rendered"
STATIC_DIR: Final = AXIOM_DIR / "static"
TEMPLATE_DIR: Final = AXIOM_DIR / "templates"

DEFAULT_BRAIN_FILE: Final = BRAIN_DIR / "my_agent_brain.json"
DEFAULT_STATE_FILE: Final = BRAIN_DIR / "my_agent_state.json"
DEFAULT_CACHE_FILE: Final = BRAIN_DIR / "interpreter_cache.json"
MODEL_VERSION_FILE: Final = BRAIN_DIR / "model_version.txt"
DEFAULT_MODEL_NAME: Final = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEFAULT_LLM_PATH: Final = MODELS_DIR / DEFAULT_MODEL_NAME
