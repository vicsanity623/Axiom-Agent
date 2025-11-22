# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- 1. LOCATE SPACY MODEL ROBUSTLY ---
import en_core_web_sm
# This gets the absolute path to the model folder on your disk
model_path = Path(en_core_web_sm.__path__[0])

# --- 2. SETUP PATHS ---
# We assume we are running this from the project root
PROJECT_ROOT = Path('.').resolve()
SRC_ROOT = PROJECT_ROOT / "src"

# --- 3. DATA FILES ---
# Format: (Source Path, Destination Path in Bundle)
datas = [
    # Map static assets to 'axiom/static' to match config.py
    (str(SRC_ROOT / "axiom" / "static"), "axiom/static"),
    (str(SRC_ROOT / "axiom" / "templates"), "axiom/templates"),
    
    # Config files
    (str(PROJECT_ROOT / "log_config.yaml"), "."),
    
    # SpaCy Model: Copy from site-packages to 'en_core_web_sm' in bundle root
    (str(model_path), "en_core_web_sm"),

    # Llama CPP Shared Libraries
    (str(PROJECT_ROOT / ".venv/lib/python3.11/site-packages/llama_cpp/lib"), "llama_cpp/lib"),

    # Models
    (str(PROJECT_ROOT / "models"), "models"),
]

# Add PyQt6 WebEngine resources
datas += collect_data_files('PyQt6.QtWebEngineCore')

# --- 4. HIDDEN IMPORTS ---
# Force PyInstaller to see these modules
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.websockets',
    'uvicorn.lifespan.on',
    'uvicorn.config',
    'spacy.lang.en',
    'srsly.msgpack.util',
    'blis',
    'cymem',
    'preshed',
    'murmurhash',
    'engineio.async_drivers.threading', # Needed for socket.io/terminado
]

# --- 5. BUILD PROCESS ---
a = Analysis(
    ['src/axiom/scripts/studio.py'],
    pathex=[str(PROJECT_ROOT), str(SRC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Axiom Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Hide terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(SRC_ROOT / "axiom" / "static" / "Axiom_Icon.icns")
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Axiom Studio',
)

app = BUNDLE(
    coll,
    name='Axiom Studio.app',
    icon=str(SRC_ROOT / "axiom" / "static" / "Axiom_Icon.icns"),
    bundle_identifier='com.axiom.studio',
)