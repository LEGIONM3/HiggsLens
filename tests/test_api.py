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
    assert data["prediction"]["model_id"] == "dummy_prior"


def test_job_state_transitions(monkeypatch):
    import time

    from backend.app.experiments.job_manager import job_manager
    from backend.app.schemas.experiments import TrainingRequest

    recorded_states = ["queued"]
    orig_update_job = job_manager._update_job

    def spy_update_job(job_id, state, message, **kwargs):
        recorded_states.append(state)
        return orig_update_job(job_id, state, message, **kwargs)

    monkeypatch.setattr(job_manager, "_update_job", spy_update_job)

    req = TrainingRequest(mode="fast", feature_set="primary_only", models=["dummy_prior"], seeds=[42])
    job = job_manager.create_job(req)

    # Wait briefly for background thread to complete
    for _ in range(50):
        if job_manager.get_job_status(job.job_id).state in ("completed", "failed"):
            break
        time.sleep(0.1)

    expected_states = [
        "queued",
        "validating_data",
        "preprocessing",
        "training",
        "evaluating",
        "saving_artifacts",
        "completed"
    ]
    # Verify every required granular transition occurs in exact chronological order
    for state in expected_states:
        assert state in recorded_states, f"Missing state transition: {state}"
    assert job_manager.get_job_status(job.job_id).state == "completed"
