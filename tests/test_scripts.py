# in tests/test_scripts.py

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import requests

from axiom.cognitive_agent import CognitiveAgent
from axiom.scripts.app import app as main_webui_app
from axiom.scripts.app_model import app as webui_app
from axiom.scripts.autonomous_trainer import main as train_main
from axiom.scripts.cnt import main as chat_main
from axiom.scripts.download_model import main as download_main
from axiom.scripts.render_model import main as render_main


def test_render_model_script(monkeypatch, tmp_path: Path):
    """
    Tests the render_model.py script by checking its file system side effects.
    """
    # We patch the constants in the namespace where they are actually USED,
    # which is inside the render_model script itself.
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

    # The rest of the setup is still correct.
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / "my_agent_brain.json").write_text(
        '{"nodes": [], "links": []}',
    )
    (tmp_path / "brain" / "interpreter_cache.json").write_text("{}")

    # Now, when render_main() runs, it will use our patched paths.
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
        # The __enter__ method is called when the 'with' block starts.
        # It should return the object that the 'as' variable will hold.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # The __exit__ method is called when the 'with' block ends.
        # We don't need to do any cleanup, so we can just pass.
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
    # We MUST mock CognitiveAgent so it doesn't try to load a real LLM.
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

    # Assert that the input mock was called for our inputs
    assert input_mock.call_count >= 2

    print("✅ cnt.py script test passed.")


def test_train_script_initialization(monkeypatch):
    """
    Tests that the autonomous_trainer.py script correctly initializes all components
    and can start its main loop, then gracefully exits for the test.
    """
    # 1. GIVEN: We mock all the heavy dependencies that the script initializes.
    monkeypatch.setattr("axiom.scripts.autonomous_trainer.CognitiveAgent", MagicMock())
    monkeypatch.setattr(
        "axiom.scripts.autonomous_trainer.KnowledgeHarvester",
        MagicMock(),
    )

    # We replace the entire BackgroundScheduler class with a mock.
    mock_scheduler_class = MagicMock()
    mock_scheduler_instance = mock_scheduler_class.return_value

    # We configure the 'start' method of our mock scheduler instance.
    # When called, it will raise a KeyboardInterrupt, simulating a user stopping the script.
    mock_scheduler_instance.start.side_effect = KeyboardInterrupt

    monkeypatch.setattr(
        "axiom.scripts.autonomous_trainer.BackgroundScheduler",
        mock_scheduler_class,
    )

    # 2. WHEN: We call the main function of the script.
    # It will run all the setup, call scheduler.start(), trigger our exception,
    # and then exit the try/except block cleanly.
    train_main()

    # 3. THEN: We assert that the scheduler's 'start' method was called exactly once.
    # This proves that the script's entire setup logic completed successfully
    # before our simulated shutdown.
    mock_scheduler_instance.start.assert_called_once()

    # We can also add a sanity check that jobs were added to the scheduler.
    assert mock_scheduler_instance.add_job.call_count > 0

    print("✅ autonomous_trainer.py script initialization test passed instantly.")


def test_webui_model_script_endpoints(monkeypatch):
    """
    Tests the Flask endpoints in app_model.py by mocking the model loading
    and agent initialization process.
    """
    # 1. GIVEN: We mock all the functions that have external dependencies (file I/O, agent creation).

    # Mock find_latest_model to return a fake path. It doesn't need to exist.
    fake_model_path = Path("rendered/fake_model.axm")
    monkeypatch.setattr(
        "axiom.scripts.app_model.find_latest_model",
        lambda *args: fake_model_path,
    )

    # Mock load_axiom_model to return fake, empty brain and cache data.
    fake_brain_data = {"nodes": [], "links": []}
    fake_cache_data = {"interpretations": [], "synthesis": []}
    monkeypatch.setattr(
        "axiom.scripts.app_model.load_axiom_model",
        lambda *args: (fake_brain_data, fake_cache_data),
    )

    # Finally, mock the CognitiveAgent class. When the script tries to create it,
    # it will get our spy object.
    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = "Hello from the mocked agent."
    monkeypatch.setattr(
        "axiom.scripts.app_model.CognitiveAgent",
        lambda *args, **kwargs: mock_agent_instance,
    )

    # We also need to prevent app.run() from starting a real, blocking web server.
    # We will replace it with a simple print statement for the test.
    monkeypatch.setattr(
        "axiom.scripts.app_model.app.run",
        lambda *args, **kwargs: print("Mock app.run() called."),
    )

    # 2. WHEN: We call the main() function to initialize the app and the global 'axiom_agent'.
    # This will now run instantly and successfully.
    from axiom.scripts.app_model import main as app_model_main

    app_model_main()

    # Get a test client from the now-initialized Flask app.
    client = webui_app.test_client()

    # 3. THEN: We send a fake POST request and verify the response.
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    json_data = response.get_json()
    assert "response" in json_data
    assert "Hello from the mocked agent." in json_data["response"]

    # Verify that the chat method on our mock was called.
    mock_agent_instance.chat.assert_called_once_with("hello")

    print("✅ app_model.py script endpoint test passed.")


def test_main_webui_script_endpoints(monkeypatch):
    """
    Tests the main Flask endpoints in app.py, including the loading sequence.
    """
    # We must use a spec'd mock to prevent errors in dependent classes.
    mock_agent_instance = MagicMock(spec=CognitiveAgent)
    mock_agent_instance.chat.return_value = "Hello from the app.py mock agent."
    mock_scheduler = MagicMock(name="MockScheduler")
    mock_cycle_mgr = MagicMock(name="MockCycleManager")

    monkeypatch.setattr(
        "axiom.scripts.app.CognitiveAgent",
        lambda *args, **kwargs: mock_agent_instance,
    )

    monkeypatch.setattr("axiom.scripts.app.KnowledgeHarvester", MagicMock())
    monkeypatch.setattr(
        "axiom.scripts.app.BackgroundScheduler",
        lambda *a, **kw: mock_scheduler,
    )
    monkeypatch.setattr(
        "axiom.scripts.app.CycleManager",
        lambda *a, **kw: mock_cycle_mgr,
    )
    monkeypatch.setattr("axiom.scripts.app.time.sleep", lambda seconds: None)

    client = main_webui_app.test_client()
    monkeypatch.setattr("axiom.scripts.app.agent_status", "uninitialized")
    monkeypatch.setattr("axiom.scripts.app.axiom_agent", None)

    response_status1 = client.get("/status")
    assert response_status1.status_code == 200
    assert response_status1.get_json()["status"] == "loading"

    monkeypatch.setattr("axiom.scripts.app.agent_status", "ready")
    monkeypatch.setattr("axiom.scripts.app.axiom_agent", mock_agent_instance)

    response_status2 = client.get("/status")
    assert response_status2.status_code == 200
    assert response_status2.get_json()["status"] == "ready"

    response_chat = client.post("/chat", json={"message": "hello"})
    assert response_chat.status_code == 200
    json_data = response_chat.get_json()
    assert "response" in json_data
    assert "Hello from the app.py mock agent." in json_data["response"]

    mock_agent_instance.chat.assert_called_once_with("hello")

    print("✅ app.py script endpoints and loading sequence test passed.")
