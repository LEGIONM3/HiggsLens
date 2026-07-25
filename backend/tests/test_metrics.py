def test_threshold_curve_happy_path(client):
    resp = client.get("/api/v1/metrics/random_forest/thresholds")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "random_forest"
    assert data["optimal_threshold"] == 0.6862
    assert isinstance(data["points"], list)
    assert len(data["points"]) > 0


def test_threshold_curve_point_structure(client):
    resp = client.get("/api/v1/metrics/random_forest/thresholds")
    pt = resp.json()["points"][0]
    assert "threshold" in pt
    assert "ams" in pt
    assert "precision" in pt
    assert "recall" in pt
    assert "f1" in pt


def test_threshold_curve_unknown_model_404(client):
    resp = client.get("/api/v1/metrics/unknown_model_abc/thresholds")
    assert resp.status_code == 404
    assert "Unknown model_id" in resp.json()["detail"]
