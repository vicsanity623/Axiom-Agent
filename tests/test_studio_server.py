from fastapi.testclient import TestClient

from axiom.scripts.studio_server import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    # Should redirect to /studio
    assert (
        response.status_code == 200
    )  # RedirectResponse follows by default? No, TestClient follows redirects.
    # Actually / returns RedirectResponse, TestClient follows it to /studio which returns 200 OK (HTML)
    assert response.status_code == 200


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data


def test_agent_settings_endpoint():
    response = client.get("/api/agent_settings")
    assert response.status_code == 200
    # It might return defaults if file doesn't exist
    data = response.json()
    assert "temperature" in data
