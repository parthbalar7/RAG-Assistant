"""Test authentication endpoints."""

import uuid


def _unique_user():
    return f"user_{uuid.uuid4().hex[:8]}"


def test_register_new_user(client):
    username = _unique_user()
    r = client.post("/api/auth/register", json={"username": username, "password": "pass1234"})
    assert r.status_code == 200
    data = r.json()
    assert "token" in data
    assert data["user"]["username"] == username


def test_register_duplicate_username(client):
    username = _unique_user()
    client.post("/api/auth/register", json={"username": username, "password": "pass1234"})
    r = client.post("/api/auth/register", json={"username": username, "password": "pass1234"})
    assert r.status_code == 400
    assert "exists" in r.json()["detail"].lower()


def test_register_short_username(client):
    r = client.post("/api/auth/register", json={"username": "ab", "password": "pass1234"})
    assert r.status_code == 422  # Pydantic validation error


def test_register_short_password(client):
    r = client.post("/api/auth/register", json={"username": _unique_user(), "password": "abc"})
    assert r.status_code == 422


def test_login_correct_credentials(client):
    username = _unique_user()
    client.post("/api/auth/register", json={"username": username, "password": "pass1234"})
    r = client.post("/api/auth/login", json={"username": username, "password": "pass1234"})
    assert r.status_code == 200
    assert "token" in r.json()


def test_login_wrong_password(client):
    username = _unique_user()
    client.post("/api/auth/register", json={"username": username, "password": "pass1234"})
    r = client.post("/api/auth/login", json={"username": username, "password": "wrongpass"})
    assert r.status_code == 401


def test_login_nonexistent_user(client):
    r = client.post("/api/auth/login", json={"username": "nobody_here", "password": "pass1234"})
    assert r.status_code == 401


def test_me_with_valid_token(client, auth_token):
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200
    assert r.json()["user"] is not None
    assert "username" in r.json()["user"]


def test_me_without_token(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    assert r.json()["user"] is None


def test_me_with_invalid_token(client):
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert r.status_code == 200
    assert r.json()["user"] is None
