from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import requests

from axiom.scripts.cnt import main as chat_main
from axiom.scripts.download_model import main as download_main
from axiom.scripts.render_model import main as render_main

if TYPE_CHECKING:
    from pathlib import Path


def test_render_model_script(monkeypatch, tmp_path: Path):
    """
    Tests the render_model.py script by checking its file system side effects.
    """
    monkeypatch.setattr("axiom.scripts.render_model.BRAIN_DIR", tmp_path / "brain")
    monkeypatch.setattr(
        "axiom.scripts.render_model.RENDERED_DIR",
        tmp_path / "rendered",
    )
    monkeypatch.setattr(
        "axiom.scripts.render_model.MODEL_VERSION_FILE",
        tmp_path / "brain" / "model_version.txt",
    )
    monkeypatch.setattr(
        "axiom.scripts.render_model.DEFAULT_BRAIN_FILE",
        tmp_path / "brain" / "my_agent_brain.json",
    )
    monkeypatch.setattr(
        "axiom.scripts.render_model.DEFAULT_CACHE_FILE",
        tmp_path / "brain" / "interpreter_cache.json",
    )

    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / "my_agent_brain.json").write_text(
        '{"nodes": [], "links": []}',
    )
    (tmp_path / "brain" / "interpreter_cache.json").write_text("{}")

    render_main()

    rendered_folder = tmp_path / "rendered"
    assert rendered_folder.exists()

    expected_file = rendered_folder / "Axiom_0.1.axm"
    assert expected_file.exists()

    with zipfile.ZipFile(expected_file, "r") as zf:
        assert "brain.json" in zf.namelist()
        assert "cache.json" in zf.namelist()
        assert "version.json" in zf.namelist()

    print("✅ render_model.py script test passed.")


class MockResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self._content = content
        self.status_code = status_code
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.RequestException("Mocked HTTP Error")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_download_model_script(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    mock_content = b"This is a fake GGUF model file."
    mock_response = MockResponse(content=mock_content)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)
    download_main()
    model_file = tmp_path / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    assert model_file.exists()
    assert model_file.read_bytes() == mock_content
    print("✅ download_model.py script test passed.")


def test_chat_script_loop(monkeypatch):
    monkeypatch.setattr("axiom.scripts.cnt.CognitiveAgent", MagicMock())

    inputs = ["hello", "show all facts"]
    input_mock = MagicMock(side_effect=inputs + [EOFError])
    monkeypatch.setattr("builtins.input", input_mock)

    print_mock = MagicMock()
    monkeypatch.setattr("builtins.print", print_mock)

    try:
        chat_main()
    except EOFError:
        pass

    assert input_mock.call_count >= 2

    print("✅ cnt.py script test passed.")
