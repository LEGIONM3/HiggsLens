def test_list_models_happy_path(client):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    model_ids = [m["model_id"] for m in data["models"]]
    assert "random_forest" in model_ids


def test_list_models_headline_metrics(client):
    resp = client.get("/api/v1/models")
    data = resp.json()
    rf = next(m for m in data["models"] if m["model_id"] == "random_forest")
    assert abs(rf["roc_auc"] - 0.8851003551263907) < 1e-4
    assert abs(rf["ams_score"] - 1.0510671984115052) < 1e-4
    assert rf["optimal_threshold"] == 0.6862


def test_get_model_metrics_happy_path(client):
    resp = client.get("/api/v1/models/random_forest/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "random_forest"
    assert data["validation_rows"] == 100000


def test_get_model_metrics_unknown_model_404(client):
    resp = client.get("/api/v1/models/non_existent_model/metrics")
    assert resp.status_code == 404
    assert "Unknown model_id" in resp.json()["detail"]
