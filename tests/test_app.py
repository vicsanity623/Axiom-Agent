from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from axiom.scripts import app as flask_app

if TYPE_CHECKING:
    from collections.abc import Generator

    from flask.testing import FlaskClient


@pytest.fixture
def client() -> Generator[FlaskClient, None, None]:
    """
    A fixture that provides a Flask test client and isolates the global state
    of the app module for each test.
    """
    flask_app.app.config.update({"TESTING": True})
    test_client = flask_app.app.test_client()

    original_agent = flask_app.axiom_agent
    original_status = flask_app.agent_status
    flask_app.axiom_agent = None
    flask_app.agent_status = "uninitialized"

    yield test_client

    flask_app.axiom_agent = original_agent
    flask_app.agent_status = original_status


def test_index_route(client: FlaskClient) -> None:
    """
    Given: The Flask app.
    When: A GET request is made to the root URL ("/").
    Then: The response should be successful and contain the main page content.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"Axiom Agent" in response.data


def test_static_file_routes(client: FlaskClient) -> None:
    """
    Given: The Flask app.
    When: Requests are made to static file routes.
    Then: The responses should be successful with the correct content type.
    """
    response = client.get("/manifest.json")
    assert response.status_code == 200
    assert response.content_type == "application/json"

    response = client.get("/sw.js")
    assert response.status_code == 200
    assert response.content_type.startswith("text/javascript")


def test_status_endpoint_initializes_agent(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The agent is uninitialized.
    When: A GET request is made to "/status".
    Then: The status should change to "loading" and agent loading should be triggered.
    """
    mock_load_agent = MagicMock()
    monkeypatch.setattr(flask_app, "load_agent", mock_load_agent)
    mock_thread = MagicMock()
    monkeypatch.setattr(threading, "Thread", mock_thread)

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json is not None
    assert response.json["status"] == "loading"
    mock_thread.assert_called_once_with(target=mock_load_agent)


def test_status_endpoint_while_loading(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The agent is already loading.
    When: A GET request is made to "/status".
    Then: The status should remain "loading" and no new thread should be started.
    """
    monkeypatch.setattr(flask_app, "agent_status", "loading")
    mock_thread = MagicMock()
    monkeypatch.setattr(threading, "Thread", mock_thread)

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json is not None
    assert response.json["status"] == "loading"
    mock_thread.assert_not_called()


def test_chat_endpoint_success(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The agent is ready.
    When: A valid POST request is made to "/chat".
    Then: The agent's chat method is called and the response is returned.
    """
    mock_agent = MagicMock()
    mock_agent.conversation_manager.chat.return_value = "Hello from the mock agent."

    monkeypatch.setattr(flask_app, "axiom_agent", mock_agent)
    monkeypatch.setattr(flask_app, "agent_status", "ready")

    response = client.post("/chat", json={"message": "test message"})

    assert response.status_code == 200
    assert response.json is not None
    assert response.json["response"] == "Hello from the mock agent."
    mock_agent.conversation_manager.chat.assert_called_once_with("test message")


def test_chat_endpoint_agent_error_state(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The agent has failed to load.
    When: A POST request is made to "/chat".
    Then: A 500 error should be returned.
    """
    monkeypatch.setattr(flask_app, "agent_status", "error: test failure")

    response = client.post("/chat", json={"message": "test message"})

    assert response.status_code == 500
    assert response.json is not None
    assert "Agent failed to load" in response.json["error"]


def test_chat_endpoint_timeout(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The agent is stuck in a "loading" state.
    When: A POST request is made to "/chat".
    Then: A 503 timeout error should be returned after a delay.
    """
    monkeypatch.setattr(flask_app, "agent_status", "loading")

    start_time = time.time()
    time_sequence = [start_time, start_time + 301]
    mock_time = MagicMock(side_effect=time_sequence)
    monkeypatch.setattr(time, "time", mock_time)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    response = client.post("/chat", json={"message": "test message"})

    assert response.status_code == 503
    assert response.json is not None
    assert "Agent is taking too long to initialize" in response.json["error"]


def test_chat_endpoint_unsupported_media_type(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The agent is ready.
    When: A POST request is made with the wrong content type.
    Then: A 415 Unsupported Media Type error should be returned.
    """
    monkeypatch.setattr(flask_app, "axiom_agent", MagicMock())
    monkeypatch.setattr(flask_app, "agent_status", "ready")

    response = client.post("/chat", data="not json", content_type="text/plain")

    assert response.status_code == 415
    assert response.data is not None


@pytest.mark.parametrize(
    ("data", "expected_error"),
    [
        # --- FIX: Send empty data with the correct content type to test the `request.json is None` branch ---
        ("", "Request must be a JSON"),
        ('{"wrong_key": "value"}', "No message provided"),
    ],
)
def test_chat_endpoint_bad_json_payload(
    client: FlaskClient,
    monkeypatch: pytest.MonkeyPatch,
    data: str,
    expected_error: str,
) -> None:
    """
    Given: The agent is ready.
    When: A POST request is made with a bad JSON payload.
    Then: A 400 Bad Request error should be returned.
    """
    monkeypatch.setattr(flask_app, "axiom_agent", MagicMock())
    monkeypatch.setattr(flask_app, "agent_status", "ready")

    # --- FIX: Send the request with the correct content type but bad data ---
    response = client.post("/chat", data=data, content_type="application/json")

    assert response.status_code == 400
    assert response.json is not None
    assert expected_error in response.json["error"]


def test_chat_endpoint_internal_exception(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: The agent's chat method raises an unexpected exception.
    When: A POST request is made to "/chat".
    Then: A 500 internal server error should be returned.
    """
    mock_agent = MagicMock()
    mock_agent.conversation_manager.chat.side_effect = ValueError("Something broke")

    monkeypatch.setattr(flask_app, "axiom_agent", mock_agent)
    monkeypatch.setattr(flask_app, "agent_status", "ready")

    response = client.post("/chat", json={"message": "test message"})

    assert response.status_code == 500
    assert response.json is not None
    assert "An internal error occurred" in response.json["error"]
