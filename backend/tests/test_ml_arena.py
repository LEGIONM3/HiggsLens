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
