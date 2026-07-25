def test_list_models_happy_path(client):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    # R004: 9 certified model artifacts are now registered (up from 5 pre-R004)
    assert len(data["models"]) >= 5
    model_ids = [m["model_id"] for m in data["models"]]
    assert "random_forest" in model_ids


def test_list_models_headline_metrics(client):
    resp = client.get("/api/v1/models")
    data = resp.json()
    rf = next(m for m in data["models"] if m["model_id"] == "random_forest")
    # R004 GPU-first canonical benchmark: RF test AUC on KaggleSet 'v' (450k events)
    assert abs(rf["roc_auc"] - 0.9061) < 1e-3
    assert rf["ams_score"] > 3.4  # R004: 3.4880


def test_get_model_metrics_happy_path(client):
    resp = client.get("/api/v1/models/random_forest/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "random_forest"
    assert "roc_auc_mean" in data  # R004: metrics.json uses roc_auc_mean key


def test_get_model_metrics_unknown_model_404(client):
    resp = client.get("/api/v1/models/non_existent_model/metrics")
    assert resp.status_code == 404
    assert "Unknown model_id" in resp.json()["detail"]
