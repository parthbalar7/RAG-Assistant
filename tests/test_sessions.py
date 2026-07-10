"""Test session CRUD endpoints."""


def test_create_session(client, auth_headers):
    r = client.post("/api/sessions", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["title"] == "New Chat"


def test_list_sessions(client, auth_headers):
    # Create two sessions
    client.post("/api/sessions", headers=auth_headers)
    client.post("/api/sessions", headers=auth_headers)
    r = client.get("/api/sessions", headers=auth_headers)
    assert r.status_code == 200
    assert len(r.json()["sessions"]) >= 2


def test_get_messages_empty_session(client, auth_headers):
    s = client.post("/api/sessions", headers=auth_headers).json()
    r = client.get(f"/api/sessions/{s['id']}/messages", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["messages"] == []


def test_rename_session(client, auth_headers):
    s = client.post("/api/sessions", headers=auth_headers).json()
    r = client.put(f"/api/sessions/{s['id']}", json={"title": "My Renamed Chat"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_delete_session(client, auth_headers):
    s = client.post("/api/sessions", headers=auth_headers).json()
    r = client.delete(f"/api/sessions/{s['id']}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"


def test_create_session_requires_auth(client):
    r = client.post("/api/sessions")
    assert r.status_code == 401


def test_list_sessions_requires_auth(client):
    r = client.get("/api/sessions")
    assert r.status_code == 401
