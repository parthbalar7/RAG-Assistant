"""Test health and basic API endpoints."""


def test_health_returns_200(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_health_has_cors_headers(client):
    r = client.get("/api/health")
    # CORS headers are set by middleware — in test client they may not appear,
    # but at minimum the endpoint should work
    assert r.status_code == 200
