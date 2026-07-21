import pytest

from backend.app.data.pipeline import DatasetPipeline
from backend.app.models.registry import model_registry


def test_model_registry_list():
    models = model_registry.list_models()
    assert "dummy_prior" in models
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "histogram_gradient_boosting" in models
    assert "xgboost" in models
    assert "mlp" in models


@pytest.mark.parametrize("model_id", [
    "dummy_prior", "logistic_regression", "random_forest", "histogram_gradient_boosting"
])
def test_required_model_candidates_fit_predict(model_id, mock_raw_df):
    cand = model_registry.create_fresh_candidate(model_id)
    assert cand.check_dependency() == "available"

    y = DatasetPipeline.encode_labels(mock_raw_df)
    cand.fit(mock_raw_df, y, feature_set="primary_only")
    assert cand.is_fitted

    probs = cand.predict_proba(mock_raw_df)
    assert probs.shape == (len(mock_raw_df), 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_optional_plugins_inspect_status():
    xgb_cand = model_registry.create_fresh_candidate("xgboost")
    status_xgb = xgb_cand.check_dependency()
    assert status_xgb in ["available", "unavailable", "incompatible"]

    mlp_cand = model_registry.create_fresh_candidate("mlp")
    status_mlp = mlp_cand.check_dependency()
    assert status_mlp in ["available", "unavailable", "incompatible"]
