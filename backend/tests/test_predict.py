def test_predict_happy_path(client, mock_feature_vector):
    payload = {
        "model_id": "random_forest",
        "features": mock_feature_vector
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "signal_probability" in data
    assert data["predicted_label"] in (0, 1)
    assert data["threshold_used"] == 0.6862
    assert data["model_id"] == "random_forest"
    assert "manifest" in data


def test_predict_custom_threshold(client, mock_feature_vector):
    payload = {
        "model_id": "random_forest",
        "features": mock_feature_vector,
        "threshold": 0.50
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["threshold_used"] == 0.50


def test_predict_missing_feature_422(client, mock_feature_vector):
    incomplete_features = dict(mock_feature_vector)
    del incomplete_features["PRI_tau_pt"]

    payload = {
        "model_id": "random_forest",
        "features": incomplete_features
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422
    assert "Missing required feature" in resp.json()["detail"]


def test_predict_non_numeric_feature_422(client, mock_feature_vector):
    invalid_features = dict(mock_feature_vector)
    invalid_features["PRI_tau_pt"] = "invalid_string"  # type: ignore

    payload = {
        "model_id": "random_forest",
        "features": invalid_features
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_unknown_model_404(client, mock_feature_vector):
    payload = {
        "model_id": "unknown_model_xyz",
        "features": mock_feature_vector
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 404
    assert "Unknown model_id" in resp.json()["detail"]


def test_predict_missing_weights_artifact_503(client, mock_feature_vector):
    payload = {
        "model_id": "corrupt_model",
        "features": mock_feature_vector
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 503
    assert "Artifact missing or corrupt" in resp.json()["detail"]


def test_predict_sentinel_minus_999_valid(client, mock_feature_vector):
    # Ensure -999.0 is accepted without 422 validation failure
    sentinel_features = dict(mock_feature_vector)
    sentinel_features["PRI_jet_subleading_pt"] = -999.0
    sentinel_features["DER_mass_MMC"] = -999.0

    payload = {
        "model_id": "random_forest",
        "features": sentinel_features
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
