import json
from pathlib import Path

import pytest
from ml.models.factory import build_model
from ml.models.registry import MODEL_SPECS, get_model_spec, list_model_specs
from ml.models.spec import DependencyMissingError


def test_registered_model_specs_count():
    specs = list_model_specs()
    assert len(specs) == 12
    expected_ids = {
        "dummy_prior", "logistic_regression", "random_forest",
        "histogram_gradient_boosting", "mlp", "mlp_torch", "xgboost", "lightgbm",
        "svm_rbf", "calibrated_ensemble", "quantum_kernel_svm",
        "variational_quantum_classifier"
    }
    assert set(specs.keys()) == expected_ids


def test_baseline_parity_with_served_artifacts():
    artifacts_dir = Path("models/artifacts")
    baseline_ids = ["dummy_prior", "logistic_regression", "random_forest", "histogram_gradient_boosting", "mlp"]

    for model_id in baseline_ids:
        spec = get_model_spec(model_id)
        assert spec is not None
        assert spec.model_id == model_id

        manifest_file = artifacts_dir / model_id / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r") as f:
                manifest = json.load(f)
            assert manifest["model_id"] == model_id
            assert manifest["random_seed"] == 42


def test_build_model_factory_and_missing_dependencies():
    for model_id, spec in MODEL_SPECS.items():
        try:
            model_obj = build_model(model_id, feature_set="all_physics")
            assert model_obj is not None
            assert hasattr(model_obj, "fit") or hasattr(model_obj, "predict")
        except DependencyMissingError as e:
            # Missing optional dependency must skip cleanly with clear exception info
            assert e.required_pkg in spec.requires
            pytest.skip(f"Optional dependency '{e.required_pkg}' for '{model_id}' is not installed.")


def test_experimental_qml_stubs_disclaimer():
    q_spec1 = get_model_spec("quantum_kernel_svm")
    assert q_spec1.experimental is True
    assert "statistical" in q_spec1.description.lower()
    assert "never quantum simulation" in q_spec1.description.lower()

    q_spec2 = get_model_spec("variational_quantum_classifier")
    assert q_spec2.experimental is True
    assert "statistical" in q_spec2.description.lower()
    assert "never quantum simulation" in q_spec2.description.lower()


def test_synthetic_fit_on_small_fixture():
    # Verify builders produce runnable pipelines on tiny synthetic fixture (<200 rows)
    import numpy as np
    import pandas as pd
    from ml.data.feature_sets import ALL_PHYSICS_FEATURES

    X_tiny = pd.DataFrame(np.random.randn(50, len(ALL_PHYSICS_FEATURES)), columns=ALL_PHYSICS_FEATURES)
    y_tiny = np.random.randint(0, 2, size=50)

    for model_id in ["dummy_prior", "logistic_regression", "random_forest", "histogram_gradient_boosting", "mlp"]:
        model = build_model(model_id)
        model.fit(X_tiny, y_tiny)
        probs = model.predict_proba(X_tiny)
        assert probs.shape == (50, 2)


def test_r005_promoted_configs_baseline_parity():
    # R005 Baseline Parity Verification for Certified Models & Promoted R005 Configurations (2026-07-26)
    # Checks that certified artifact metrics in models/artifacts/ match expected benchmark values,
    # and validates that R005 uplift report (reports/r005_uplift_2026-07-26.md) reflects promoted AUC gains.
    artifacts_dir = Path("models/artifacts")
    certified_expectations = {
        "xgboost": 0.9096,
        "lightgbm": 0.9087,
        "random_forest": 0.9061,
        "mlp_torch": 0.9064,
        "svm_rbf": 0.8723,
    }

    for model_id, expected_auc in certified_expectations.items():
        metrics_file = artifacts_dir / model_id / "metrics.json"
        assert metrics_file.exists(), f"Missing certified metrics for {model_id}"
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert abs(metrics["roc_auc_mean"] - expected_auc) < 0.005, (
            f"Certified artifact parity check failed for {model_id}: expected ~{expected_auc}, got {metrics['roc_auc_mean']}"
        )

    # Confirm R005 uplift report exists and documents promoted gains
    uplift_report = Path("reports/r005_uplift_2026-07-26.md")
    assert uplift_report.exists(), "R005 uplift report missing!"
    report_text = uplift_report.read_text(encoding="utf-8")
    assert "0.9123" in report_text, "R005 promoted XGBoost ROC-AUC (0.9123) missing from uplift report!"
    assert "0.8868" in report_text, "R005 promoted SVM RBF ROC-AUC (0.8868) missing from uplift report!"


