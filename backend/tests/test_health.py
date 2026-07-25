def test_health_endpoint_happy_path(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["available_models_count"] >= 1


def test_health_endpoint_structure(client):
    resp = client.get("/health")
    data = resp.json()
    assert isinstance(data["available_models_count"], int)
    assert data["version"] == "0.1.0"
