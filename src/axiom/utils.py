from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def atomic_write(path: Path, data: bytes) -> None:
    """Write data to a file atomically.

    Writes to a temporary file in the same directory and then renames it to
    the final destination. This prevents file corruption from partial writes.

    Args:
        path: The final destination path for the file.
        data: The data (in bytes) to be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent)
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (color codes) from a string."""
    return ANSI_ESCAPE_PATTERN.sub("", text)
