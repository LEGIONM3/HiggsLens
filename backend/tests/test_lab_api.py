import io
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from backend.app.core.config import settings
from backend.app.main import create_app
from backend.app.services.lab.job_runner import lab_job_runner
from backend.app.services.model_registry import model_registry_service
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_csv_bytes():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "PRI_tau_pt": np.random.uniform(10, 100, size=n),
        "PRI_tau_eta": np.random.uniform(-2.5, 2.5, size=n),
        "PRI_lep_pt": np.random.uniform(10, 80, size=n),
        "PRI_met": np.random.uniform(0, 150, size=n),
        "Label": np.random.choice(["s", "b"], size=n),
        "Weight": np.random.uniform(0.1, 2.0, size=n),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_lab_dataset_upload_happy_path(client, sample_csv_bytes):
    response = client.post(
        "/api/v1/lab/datasets",
        data={
            "feature_columns": "PRI_tau_pt, PRI_tau_eta, PRI_lep_pt, PRI_met",
            "label_column": "Label",
            "weight_column": "Weight",
        },
        files={"file": ("my_test_data.csv", sample_csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    # Assert server-generated UUID (path-traversal protection)
    uuid_obj = uuid.UUID(data["dataset_id"])
    assert str(uuid_obj) == data["dataset_id"]
    assert data["row_count"] == 100
    assert data["feature_columns"] == ["PRI_tau_pt", "PRI_tau_eta", "PRI_lep_pt", "PRI_met"]
    assert data["label_column"] == "Label"
    assert data["weight_column"] == "Weight"
    assert len(data["content_hash"]) == 64


def test_lab_dataset_upload_error_contracts(client):
    # 1. Empty file -> 422
    res_empty = client.post(
        "/api/v1/lab/datasets",
        data={"feature_columns": "feat1", "label_column": "Label"},
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert res_empty.status_code == 422

    # 2. Missing feature column -> 422
    csv_bytes = b"feat1,Label\n1.0,s\n2.0,b\n"
    res_missing = client.post(
        "/api/v1/lab/datasets",
        data={"feature_columns": "non_existent_col", "label_column": "Label"},
        files={"file": ("data.csv", csv_bytes, "text/csv")},
    )
    assert res_missing.status_code == 422

    # 3. Non-binary label column -> 422
    csv_multi_class = b"feat1,Label\n1.0,s\n2.0,b\n3.0,c\n"
    res_multi = client.post(
        "/api/v1/lab/datasets",
        data={"feature_columns": "feat1", "label_column": "Label"},
        files={"file": ("data.csv", csv_multi_class, "text/csv")},
    )
    assert res_multi.status_code == 422


def test_lab_experiment_lifecycle_and_evaluation_flow(client, sample_csv_bytes):
    # Step 1: Upload dataset
    upload_res = client.post(
        "/api/v1/lab/datasets",
        data={
            "feature_columns": "PRI_tau_pt, PRI_tau_eta, PRI_lep_pt, PRI_met",
            "label_column": "Label",
            "weight_column": "Weight",
        },
        files={"file": ("experiment_test.csv", sample_csv_bytes, "text/csv")},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    # Step 2: Create experiment
    exp_req = {
        "dataset_id": dataset_id,
        "model_ids": ["logistic_regression", "random_forest"],
        "split_config": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "seed": 42,
    }
    exp_res = client.post("/api/v1/lab/experiments", json=exp_req)
    assert exp_res.status_code == 202
    exp_summary = exp_res.json()
    experiment_id = exp_summary["experiment_id"]
    assert exp_summary["status"] == "queued"

    # Step 3: Synchronously execute background training job
    lab_job_runner.execute_experiment_job(experiment_id)

    # Step 4: Verify experiment detail & train->val->test metrics flow
    detail_res = client.get(f"/api/v1/lab/experiments/{experiment_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["summary"]["status"] == "completed"

    results = detail["per_model_results"]
    assert "logistic_regression" in results
    assert "random_forest" in results

    lr_res = results["logistic_regression"]
    assert "test_metrics" in lr_res
    assert "validation_optimal_threshold" in lr_res
    assert lr_res["is_weighted"] is True
    assert lr_res["test_metrics"]["roc_auc_mean"] >= 0.0


def test_lab_experiment_concurrency_cap_409_conflict(client, sample_csv_bytes, tmp_path: Path):
    upload_res = client.post(
        "/api/v1/lab/datasets",
        data={
            "feature_columns": "PRI_tau_pt, PRI_tau_eta, PRI_lep_pt, PRI_met",
            "label_column": "Label",
        },
        files={"file": ("concurrency.csv", sample_csv_bytes, "text/csv")},
    )
    dataset_id = upload_res.json()["dataset_id"]

    # Enqueue first experiment and force status to "running" on disk
    exp_id1 = lab_job_runner.create_and_enqueue_job(
        dataset_id=dataset_id,
        model_ids=["logistic_regression"],
        split_config={"train": 0.7, "validation": 0.15, "test": 0.15},
    )
    manifest_path = lab_job_runner.lab_artifacts_dir / exp_id1 / "experiment_manifest.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)
    data["status"] = "running"
    with open(manifest_path, "w") as f:
        json.dump(data, f)

    # Attempt to submit second experiment -> must return 409 Conflict
    exp_req2 = {
        "dataset_id": dataset_id,
        "model_ids": ["random_forest"],
    }
    res_conflict = client.post("/api/v1/lab/experiments", json=exp_req2)
    assert res_conflict.status_code == 409
    assert "currently running" in res_conflict.json()["detail"]

    # Clean up status so subsequent tests pass cleanly
    data["status"] = "completed"
    with open(manifest_path, "w") as f:
        json.dump(data, f)


def test_certified_zone_segregation(client):
    """
    STRICT SEGREGATION ASSERTION:
    Verifies that the certified ModelRegistryService ONLY scans models/artifacts/
    and never loads lab artifacts from models/lab_artifacts/.
    """
    certified_ids = model_registry_service.list_model_ids()

    assert "dummy_prior" in certified_ids
    assert "random_forest" in certified_ids

    # Ensure no lab experiment ID is in certified model registry
    for m_id in certified_ids:
        assert not m_id.startswith("exp_")
        assert not (settings.LAB_ARTIFACTS_DIR / m_id).exists()

    # Check GET /api/v1/models response
    api_models_res = client.get("/api/v1/models")
    assert api_models_res.status_code == 200
    models_data = api_models_res.json()["models"]
    # R004: 9 certified model artifacts are now registered (up from 5 pre-R004)
    assert len(models_data) >= 5
