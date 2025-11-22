# src/axiom/config.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final


def get_project_root() -> Path:
    """
    Determines the root path for resources.
    - If Frozen (PyInstaller): Uses sys._MEIPASS (the temp folder where assets live).
    - If Dev: Uses the git repo root.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller --onefile or --onedir mode
        # Assets are mapped to the root of _MEIPASS in the .spec file
        return Path(sys._MEIPASS)

    # Development mode: src/axiom/config.py -> up 3 levels -> Repo Root
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT: Final = get_project_root()

# --- DIR DEFINITIONS ---

if getattr(sys, "frozen", False):
    # In the bundle, we map 'src/axiom/static' -> 'axiom/static'
    # So we don't use a 'src' folder intermediate.
    AXIOM_DIR = PROJECT_ROOT / "axiom"
    SRC_ROOT = PROJECT_ROOT
else:
    # In dev, we follow the source structure
    SRC_ROOT = PROJECT_ROOT / "src"
    AXIOM_DIR = SRC_ROOT / "axiom"

# These paths now align with both Dev structure AND the .spec file mapping
STATIC_DIR: Final = AXIOM_DIR / "static"
TEMPLATE_DIR: Final = AXIOM_DIR / "templates"

# --- DATA & BRAIN PATHS ---

# Allow overriding data dir via env var, otherwise default to PROJECT_ROOT
_data_dir_str = os.getenv("AXIOM_DATA_DIR", str(PROJECT_ROOT))
DATA_ROOT: Final = Path(_data_dir_str)

MODELS_DIR: Final = DATA_ROOT / "models"

# In frozen mode, we might want to write brain/state files to the user's
# document folder instead of the read-only bundle, but for now we keep
# logic consistent with your existing setup.
if getattr(sys, "frozen", False):
    # In bundle, we look for brain files at the top level or specific folder
    # tailored to how you want the app to persist data.
    # For read-only app usage, this is fine.
    BRAIN_DIR = AXIOM_DIR / "brain"
    # If you want persistence in the app, you should eventually change this to:
    # BRAIN_DIR = Path.home() / ".axiom" / "brain"
else:
    BRAIN_DIR = AXIOM_DIR / "brain"

RENDERED_DIR: Final = AXIOM_DIR / "rendered"

DEFAULT_BRAIN_FILE: Final = BRAIN_DIR / "my_agent_brain.json"
MASTER_BRAIN_FILE: Final = BRAIN_DIR / "master_brain.json"
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

# --- Neo4j Database Configuration ---
NEO4J_URI: Final[str | None] = os.getenv("NEO4J_URI")
NEO4J_USER: Final[str | None] = os.getenv("NEO4J_USER")
NEO4J_PASSWORD: Final[str | None] = os.getenv("NEO4J_PASSWORD")
