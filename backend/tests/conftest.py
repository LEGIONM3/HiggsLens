import json
from pathlib import Path
from typing import Any, Dict, Generator

import joblib
import numpy as np
import pandas as pd
import pytest
from backend.app.core.config import settings
from backend.app.main import create_app
from backend.app.models.dummy import DummyPriorCandidate
from backend.app.services.model_registry import ModelRegistryService
from fastapi.testclient import TestClient

FEATURE_NAMES = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_deltar_tau_lep",
    "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau", "DER_met_phi_centrality",
    "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
    "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met",
    "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt",
    "PRI_jet_leading_eta", "PRI_jet_leading_phi", "PRI_jet_subleading_pt", "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi", "PRI_jet_all_pt"
]


@pytest.fixture
def mock_feature_vector() -> Dict[str, float]:
    vec = {}
    for i, name in enumerate(FEATURE_NAMES):
        if name.startswith("PRI_jet") and i > 23:
            vec[name] = -999.0  # test valid missing sentinel
        else:
            vec[name] = float(10.0 + i)
    return vec


@pytest.fixture
def temp_artifacts_dir(tmp_path: Path) -> Generator[Path, None, None]:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create valid 'random_forest' dummy artifact
    rf_dir = artifacts_dir / "random_forest"
    rf_dir.mkdir()

    metrics = {
        "model_id": "random_forest",
        "feature_set": "all_physics",
        "mode": "fast",
        "seeds_evaluated": [42],
        "roc_auc_mean": 0.8851003551263907,
        "roc_auc_std": 0.0,
        "pr_auc_mean": 0.8130697402413716,
        "pr_auc_std": 0.0,
        "log_loss_mean": 0.4051373212032764,
        "log_loss_std": 0.0,
        "balanced_accuracy_mean": 0.7044525359866514,
        "balanced_accuracy_std": 0.0,
        "f1_mean": 0.5865951322668442,
        "f1_std": 0.0,
        "brier_score_mean": 0.12726847917264578,
        "brier_score_std": 0.0,
        "optimal_threshold": 0.6862,
        "ams_score": 1.0510671984115052,
        "ams_default_threshold_score": 0.9563481957102055,
        "training_duration_seconds": 0.67,
        "stability_status": "not_assessed",
        "calibration_status": "not_calibrated",
        "validation_rows": 100000,
        "precision_05": 0.78,
        "recall_05": 0.68,
        "precision_selected": 0.87,
        "recall_selected": 0.44,
        "confusion_matrix_05": {"tn": 59552, "fp": 6423, "fn": 10847, "tp": 23178},
        "confusion_matrix_selected": {"tn": 63908, "fp": 2067, "fn": 19046, "tp": 14979},
        "weighted_signal_yield_s": 33.4,
        "weighted_background_yield_b": 989.0,
        "ams_br": 10.0,
        "calibration_method": "none",
        "expected_calibration_error": 0.043,
        "reliability_bins": [
            {"bin_low": 0.0, "bin_high": 0.5, "confidence": 0.2, "accuracy": 0.1, "count": 100},
            {"bin_low": 0.5, "bin_high": 1.0, "confidence": 0.8, "accuracy": 0.7, "count": 100}
        ]
    }
    with open(rf_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    schema = {
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "sentinel_value": -999.0
    }
    with open(rf_dir / "feature_schema.json", "w") as f:
        json.dump(schema, f)

    manifest = {
        "model_id": "random_forest",
        "git_commit": "6f3555d",
        "random_seed": 42,
        "dataset_hash": "f370a6c17b2c8f552fb4620385cf8667f9943a5b1afb3f7c6ead84510d04a8dc",
        "created_at": "2026-07-21T09:05:18.717413+00:00"
    }
    with open(rf_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Fit dummy candidate model and save joblib
    cand = DummyPriorCandidate()
    X_dum = pd.DataFrame(np.zeros((10, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    y_dum = np.array([0, 1] * 5)
    cand.fit(X_dum, y_dum)
    joblib.dump(cand, rf_dir / "model.joblib")

    # 2. Create 'corrupt_model' directory without model.joblib to test 503 error contract
    corrupt_dir = artifacts_dir / "corrupt_model"
    corrupt_dir.mkdir()
    with open(corrupt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(corrupt_dir / "feature_schema.json", "w") as f:
        json.dump(schema, f)
    with open(corrupt_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    yield artifacts_dir


@pytest.fixture
def client(temp_artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", temp_artifacts_dir)
    # Re-initialize registry service with temp artifacts dir
    from backend.app.services.metrics_service import metrics_service
    from backend.app.services.model_registry import model_registry_service
    from backend.app.services.prediction_service import prediction_service

    test_registry = ModelRegistryService(temp_artifacts_dir)
    monkeypatch.setattr("backend.app.services.model_registry.model_registry_service", test_registry)
    monkeypatch.setattr(prediction_service, "registry", test_registry)
    monkeypatch.setattr(metrics_service, "registry", test_registry)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_raw_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    data: Dict[str, Any] = {
        "EventId": np.arange(100000, 100000 + n),
        "Label": np.random.choice(["s", "b"], size=n, p=[0.34, 0.66]),
        "Weight": np.random.uniform(0.001, 5.0, size=n),
        "KaggleSet": np.random.choice(["t", "b", "v", "u"], size=n, p=[0.5, 0.2, 0.2, 0.1]),
        "PRI_tau_pt": np.random.uniform(20.0, 150.0, size=n),
        "PRI_tau_eta": np.random.uniform(-2.5, 2.5, size=n),
        "PRI_tau_phi": np.random.uniform(-3.14, 3.14, size=n),
        "PRI_lep_pt": np.random.uniform(20.0, 150.0, size=n),
        "PRI_lep_eta": np.random.uniform(-2.5, 2.5, size=n),
        "PRI_lep_phi": np.random.uniform(-3.14, 3.14, size=n),
        "PRI_met": np.random.uniform(10.0, 200.0, size=n),
        "PRI_met_phi": np.random.uniform(-3.14, 3.14, size=n),
        "PRI_met_sumet": np.random.uniform(100.0, 800.0, size=n),
        "PRI_jet_num": np.random.choice([0, 1, 2, 3], size=n),
    }

    for col in FEATURE_NAMES:
        if col not in data:
            data[col] = np.random.uniform(0.0, 100.0, size=n)

    df = pd.DataFrame(data)

    mask_0 = df["PRI_jet_num"] == 0
    df.loc[mask_0, ["PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi", "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_lep_eta_centrality"]] = -999.0

    mask_1 = df["PRI_jet_num"] <= 1
    df.loc[mask_1, ["PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_lep_eta_centrality"]] = -999.0

    return df

