# hooks/hook-llama_cpp.py
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

# This collects ALL .dylib files from llama_cpp/lib (and subfolders)
binaries = collect_dynamic_libs("llama_cpp", include_py_files=False)

# This collects any non-code data files (rarely needed but safe)
datas = collect_data_files("llama_cpp")

# Some versions need the metadata for import to work
datas += copy_metadata("llama_cpp")
