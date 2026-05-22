"""
Integration test: root endpoint health check.
Uses the minimal TestClient fixture from conftest.py.
No external services required.
"""


def test_root_endpoint_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_root_endpoint_has_message_key(client):
    response = client.get("/")
    data = response.json()
    assert "message" in data


def test_root_message_mentions_automl(client):
    response = client.get("/")
    data = response.json()
    assert "AutoML" in data["message"] or "automl" in data["message"].lower()
