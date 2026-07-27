"""Smoke test for the API health endpoint."""

from app.main import app
from tests.client import TestClient

client = TestClient(app)


def test_health() -> None:
    """GET /api/health returns 200 with a status indicator."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
