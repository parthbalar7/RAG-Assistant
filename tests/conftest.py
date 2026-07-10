"""Shared test fixtures for the RAG backend."""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Point database to a temp file before importing app
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["RAG_DATABASE_PATH"] = _tmp_db.name
os.environ["RAG_JWT_SECRET"] = "test-secret-for-ci"
os.environ["RAG_ANTHROPIC_API_KEY"] = ""  # disable real API calls in tests
_tmp_db.close()

from api.server import app  # noqa: E402
from api import database as db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    """Initialize the test database once per test session."""
    db.init_db()
    yield
    # Cleanup
    try:
        os.unlink(_tmp_db.name)
    except OSError:
        pass


@pytest.fixture
def client():
    """FastAPI TestClient for HTTP endpoint testing."""
    return TestClient(app)


@pytest.fixture
def auth_token(client):
    """Register a fresh test user and return a valid JWT token."""
    import uuid
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    r = client.post("/api/auth/register", json={"username": username, "password": "testpass123"})
    assert r.status_code == 200
    return r.json()["token"]


@pytest.fixture
def auth_headers(auth_token):
    """Authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}
