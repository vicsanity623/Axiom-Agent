from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from axiom.scripts import app_model

if TYPE_CHECKING:
    from collections.abc import Generator

    from flask.testing import FlaskClient


@pytest.fixture
def client() -> Generator[FlaskClient, None, None]:
    """A test client for the Flask app."""
    app_model.app.config["TESTING"] = True
    with app_model.app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def mock_agent_global(monkeypatch):
    """Ensure the global agent is reset for each test."""
    monkeypatch.setattr(app_model, "axiom_agent", None)


def create_dummy_axm(
    directory: Path,
    filename: str,
    brain_data: dict,
    cache_data: dict | None = None,
    version_data: dict | None = None,
    is_corrupt: bool = False,
    missing_file: str | None = None,
) -> Path:
    """Helper function to create dummy .axm files for testing."""
    filepath = directory / filename
    cache_data = cache_data or {}
    version_data = version_data or {}

    if is_corrupt:
        filepath.write_text("this is not a zip file")
        return filepath

    with zipfile.ZipFile(filepath, "w") as zf:
        if missing_file != "brain.json":
            brain_bytes = json.dumps(brain_data).encode("utf-8")
            zf.writestr("brain.json", brain_bytes)
            if "checksum" not in version_data:
                version_data["checksum"] = hashlib.sha256(brain_bytes).hexdigest()

        if missing_file != "cache.json":
            zf.writestr("cache.json", json.dumps(cache_data))
        if missing_file != "version.json":
            zf.writestr("version.json", json.dumps(version_data))

    return filepath


class TestFileFunctions:
    """Tests for file handling functions like find_latest_model and load_axiom_model."""

    def test_find_latest_model_success(self, tmp_path: Path):
        """Test that the most recent model file is found correctly."""
        (tmp_path / "Axiom_0.1.axm").touch()
        time.sleep(0.1)
        latest_file = tmp_path / "Axiom_0.2.axm"
        latest_file.touch()
        time.sleep(0.1)
        (tmp_path / "other_file.txt").touch()

        found_file = app_model.find_latest_model(tmp_path)
        assert found_file == latest_file

    def test_find_latest_model_no_models(self, tmp_path: Path, caplog):
        """Test that None is returned when no model files are present."""
        (tmp_path / "other_file.txt").touch()
        with caplog.at_level(logging.CRITICAL):
            found_file = app_model.find_latest_model(tmp_path)
            assert found_file is None
            assert "No .axm model files found" in caplog.text

    def test_find_latest_model_exception(self, monkeypatch, caplog):
        """Test that an exception during file search is handled gracefully."""
        monkeypatch.setattr(
            Path, "glob", MagicMock(side_effect=OSError("Permission denied"))
        )
        with caplog.at_level(logging.CRITICAL):
            found_file = app_model.find_latest_model(Path("."))
            assert found_file is None
            assert "Error while searching for latest model" in caplog.text

    def test_load_axiom_model_success_with_checksum(self, tmp_path: Path):
        """Test loading a valid .axm file with a correct checksum."""
        axm_path = create_dummy_axm(
            tmp_path,
            "Axiom_1.0.axm",
            brain_data={"nodes": []},
            version_data={"schema_version": 1},
        )
        result = app_model.load_axiom_model(axm_path)
        assert result is not None
        brain, cache = result
        assert brain == {"nodes": []}
        assert cache == {}

    def test_load_axiom_model_file_not_found(self, tmp_path: Path, caplog):
        """Test loading a model that does not exist."""
        with caplog.at_level(logging.ERROR):
            result = app_model.load_axiom_model(tmp_path / "nonexistent.axm")
            assert result is None
            assert "model file that does not exist" in caplog.text

    def test_load_axiom_model_bad_zip(self, tmp_path: Path, caplog):
        """Test loading a corrupt (non-zip) .axm file."""
        axm_path = create_dummy_axm(tmp_path, "Axiom_1.0.axm", {}, is_corrupt=True)
        with caplog.at_level(logging.CRITICAL):
            result = app_model.load_axiom_model(axm_path)
            assert result is None
            assert "Failed to read .axm archive" in caplog.text

    def test_load_axiom_model_missing_key_file(self, tmp_path: Path, caplog):
        """Test loading a zip missing a required file like brain.json."""
        axm_path = create_dummy_axm(
            tmp_path, "Axiom_1.0.axm", {}, missing_file="brain.json"
        )
        with caplog.at_level(logging.CRITICAL):
            result = app_model.load_axiom_model(axm_path)
            assert result is None
            assert "missing required files" in caplog.text

    def test_load_axiom_model_invalid_json(self, tmp_path: Path, caplog):
        """Test loading a model with corrupt JSON inside."""
        axm_path = tmp_path / "Axiom_1.0.axm"
        with zipfile.ZipFile(axm_path, "w") as zf:
            zf.writestr("brain.json", "{not_valid_json}")
            zf.writestr("cache.json", "{}")
            zf.writestr("version.json", "{}")
        with caplog.at_level(logging.CRITICAL):
            result = app_model.load_axiom_model(axm_path)
            assert result is None
            assert "Failed to parse JSON" in caplog.text

    def test_load_axiom_model_security_risk(self, tmp_path: Path, caplog):
        """Test loading a model with a malicious path in the zip file."""
        axm_path = tmp_path / "Axiom_1.0.axm"
        with zipfile.ZipFile(axm_path, "w") as zf:
            zf.writestr("../malicious.txt", "hacked")
        with caplog.at_level(logging.CRITICAL):
            result = app_model.load_axiom_model(axm_path)
            assert result is None
            assert "Security risk: Invalid path" in caplog.text

    def test_load_axiom_model_unsupported_schema(self, tmp_path: Path, caplog):
        """Test loading a model with an unsupported schema version."""
        axm_path = create_dummy_axm(
            tmp_path, "Axiom_1.0.axm", {}, version_data={"schema_version": 99}
        )
        with caplog.at_level(logging.CRITICAL):
            result = app_model.load_axiom_model(axm_path)
            assert result is None
            assert "Unsupported schema version" in caplog.text

    def test_load_axiom_model_checksum_mismatch(self, tmp_path: Path, caplog):
        """Test loading a model where the brain checksum is incorrect."""
        axm_path = create_dummy_axm(
            tmp_path,
            "Axiom_1.0.axm",
            {"nodes": []},
            version_data={"schema_version": 1, "checksum": "incorrect_checksum"},
        )
        with caplog.at_level(logging.CRITICAL):
            result = app_model.load_axiom_model(axm_path)
            assert result is None
            assert "Checksum mismatch" in caplog.text

    def test_load_axiom_model_unexpected_exception(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """Test that generic exceptions during loading are caught."""
        axm_path = create_dummy_axm(tmp_path, "Axiom_1.0.axm", {"nodes": []})
        monkeypatch.setattr(
            json, "loads", MagicMock(side_effect=Exception("Unexpected error"))
        )
        with caplog.at_level(logging.CRITICAL):
            result = app_model.load_axiom_model(axm_path)
            assert result is None
            assert "An unexpected error occurred" in caplog.text


class TestRoutes:
    """Tests for the Flask web application routes."""

    def test_index_route(self, client: FlaskClient, monkeypatch):
        """Test the main '/' route."""
        mock_render = MagicMock(return_value="<html></html>")
        monkeypatch.setattr(app_model, "render_template", mock_render)
        response = client.get("/")
        assert response.status_code == 200
        mock_render.assert_called_once_with("index.html")

    def test_manifest_route(self, client: FlaskClient, monkeypatch):
        """Test the '/manifest.json' route."""
        mock_send = MagicMock(return_value="{}")
        monkeypatch.setattr(app_model, "send_from_directory", mock_send)
        response = client.get("/manifest.json")
        assert response.status_code == 200
        mock_send.assert_called_once_with(str(app_model.STATIC_DIR), "manifest.json")

    def test_service_worker_route(self, client: FlaskClient, monkeypatch):
        """Test the '/sw.js' route."""
        mock_send = MagicMock(return_value="self.addEventListener('fetch', ...)")
        monkeypatch.setattr(app_model, "send_from_directory", mock_send)
        response = client.get("/sw.js")
        assert response.status_code == 200
        mock_send.assert_called_once_with(str(app_model.STATIC_DIR), "sw.js")

    def test_status_route_loading(self, client: FlaskClient):
        """Test '/status' when the agent is not yet loaded."""
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json is not None
        assert response.json["status"] == "loading_model"

    def test_status_route_ready(self, client: FlaskClient, monkeypatch):
        """Test '/status' when the agent is ready."""
        monkeypatch.setattr(app_model, "axiom_agent", MagicMock())
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json is not None
        assert response.json["status"] == "ready"

    def test_chat_route_success(self, client: FlaskClient, monkeypatch):
        """Test a successful chat request."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "Hello from Axiom."
        monkeypatch.setattr(app_model, "axiom_agent", mock_agent)

        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 200
        assert response.json is not None
        assert response.json["response"] == "Hello from Axiom."
        mock_agent.chat.assert_called_once_with("Hello")

    def test_chat_route_agent_not_ready(self, client: FlaskClient):
        """Test '/chat' when the agent is not loaded."""
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 503
        assert response.json is not None
        assert "Agent is not available" in response.json["error"]

    def test_chat_route_bad_request(self, client: FlaskClient, monkeypatch):
        """Test '/chat' with an invalid JSON payload."""
        monkeypatch.setattr(app_model, "axiom_agent", MagicMock())
        response = client.post("/chat", json={"wrong_key": "Hello"})
        assert response.status_code == 400
        assert response.json is not None
        assert "Invalid request" in response.json["error"]

    def test_chat_route_internal_error(self, client: FlaskClient, monkeypatch):
        """Test '/chat' when the agent's chat method raises an exception."""
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = Exception("Something went wrong")
        monkeypatch.setattr(app_model, "axiom_agent", mock_agent)

        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 500
        assert response.json is not None
        assert "An internal error occurred" in response.json["error"]


class TestMainFunction:
    """Tests for the main() entry point function."""

    @patch("argparse.ArgumentParser")
    def test_main_success(self, mock_argparse, monkeypatch, tmp_path):
        """Test the successful execution path of main()."""
        mock_args = MagicMock()
        mock_args.host = "127.0.0.1"
        mock_args.port = 7501
        mock_argparse.return_value.parse_args.return_value = mock_args

        mock_find = MagicMock(return_value=tmp_path / "model.axm")
        mock_load = MagicMock(return_value=({}, {}))
        mock_agent_class = MagicMock()
        mock_app_run = MagicMock()

        monkeypatch.setattr(app_model, "find_latest_model", mock_find)
        monkeypatch.setattr(app_model, "load_axiom_model", mock_load)
        monkeypatch.setattr(app_model, "CognitiveAgent", mock_agent_class)
        monkeypatch.setattr(app_model.app, "run", mock_app_run)

        result = app_model.main()

        assert result == 0
        mock_find.assert_called_once()
        mock_load.assert_called_once_with(tmp_path / "model.axm")
        mock_agent_class.assert_called_once()
        mock_app_run.assert_called_once_with(host="127.0.0.1", port=7501, debug=False)

    @patch("argparse.ArgumentParser")
    def test_main_no_model_found(self, mock_argparse, monkeypatch):
        """Test main() when find_latest_model returns None."""
        mock_argparse.return_value.parse_args.return_value = MagicMock()
        monkeypatch.setattr(
            app_model, "find_latest_model", MagicMock(return_value=None)
        )

        result = app_model.main()
        assert result == 1

    @patch("argparse.ArgumentParser")
    def test_main_model_load_fails(self, mock_argparse, monkeypatch, tmp_path):
        """Test main() when load_axiom_model returns None."""
        mock_argparse.return_value.parse_args.return_value = MagicMock()
        monkeypatch.setattr(
            app_model,
            "find_latest_model",
            MagicMock(return_value=tmp_path / "model.axm"),
        )
        monkeypatch.setattr(app_model, "load_axiom_model", MagicMock(return_value=None))

        result = app_model.main()
        assert result == 1
