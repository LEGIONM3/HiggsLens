from fastapi.testclient import TestClient

from backend.app.data.pipeline import DatasetPipeline
from backend.app.main import app
from backend.app.models.registry import model_registry

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "HiggsLens"


def test_config_endpoint():
    resp = client.get("/api/v1/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cern_record_id"] == 328
    assert data["dataset_filename"] == "atlas-higgs-challenge-2014-v2.csv.gz"


def test_dataset_status_endpoint():
    resp = client.get("/api/v1/dataset/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "exists" in data
    assert data["record_id"] == 328


def test_models_registry_endpoint():
    resp = client.get("/api/v1/models/registry")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "dummy_prior" in data["models"]
    assert "logistic_regression" in data["models"]


def test_predict_endpoint_with_custom_features(mock_raw_df):
    # Ensure dummy_prior is fitted so prediction can run without dataset on disk
    cand = model_registry.get_candidate("dummy_prior")
    y = DatasetPipeline.encode_labels(mock_raw_df)
    cand.fit(mock_raw_df, y, feature_set="primary_only")

    payload = {
        "model_id": "dummy_prior",
        "feature_set": "primary_only",
        "features": {
            "PRI_tau_pt": 45.0,
            "PRI_lep_pt": 60.0,
            "PRI_jet_num": 1.0,
            "PRI_jet_leading_pt": 80.0
        }
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "objects" in data
    assert "prediction" in data
    assert data["prediction"]["model_id"] == "dummy_prior"
