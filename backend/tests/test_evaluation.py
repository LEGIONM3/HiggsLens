import numpy as np
import pandas as pd
import pytest
from backend.app.evaluation.evaluator import ModelEvaluator
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo
from sklearn.metrics import log_loss as sk_log_loss


class MockCandidate(ModelCandidate):
    def __init__(self, probs: np.ndarray, classes: np.ndarray = np.array([0, 1])):
        super().__init__(model_id="mock", display_name="Mock", required=False, preprocessing_strategy="native", supports_missing=False)
        self.probs = probs
        self.model = type("MockSklearnModel", (), {"classes_": classes})()
        self.is_fitted = True

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters) -> "MockCandidate":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.probs

    def check_dependency(self) -> str:
        return "available"

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id="mock", display_name="Mock", status="available", required=False,
            supported_feature_sets=["all_physics"], supports_missing=False,
            preprocessing_pipeline="native", hyperparameters_schema={}
        )


def test_log_loss_not_one_and_exact_match():
    df_val = pd.DataFrame({
        "Label": ["s", "b", "b", "s"],
        "Weight": [1.0, 1.0, 1.0, 1.0],
        "EventId": [1, 2, 3, 4]
    })
    probs = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.15, 0.85]
    ])
    cand = MockCandidate(probs=probs)
    res = ModelEvaluator.evaluate_model_on_val(cand, df_val)

    y_true = np.array([1, 0, 0, 1])
    expected_ll = sk_log_loss(y_true, probs[:, 1])

    assert res["log_loss"] != 1.0000
    assert pytest.approx(res["log_loss"], rel=1e-7) == expected_ll
    assert pytest.approx(res["log_loss"], abs=1e-5) == 0.21192


def test_invalid_probabilities_fail_validation():
    df_val = pd.DataFrame({
        "Label": ["s", "b"],
        "Weight": [1.0, 1.0],
        "EventId": [1, 2]
    })
    cand = MockCandidate(probs=np.array([[-0.1, 1.1], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="outside \\[0,1\\]"):
        ModelEvaluator.evaluate_model_on_val(cand, df_val)


def test_ams_invariant_and_formula_agreement():
    from backend.app.physics.metrics import compute_ams
    df_val = pd.DataFrame({
        "Label": ["s", "b", "s", "b"],
        "Weight": [50.0, 100.0, 30.0, 200.0],
        "EventId": [1, 2, 3, 4]
    })
    probs = np.array([
        [0.1, 0.9],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.9, 0.1]
    ])
    cand = MockCandidate(probs=probs)
    res = ModelEvaluator.evaluate_model_on_val(cand, df_val, ams_br=10.0)
    expected_ams = compute_ams(res["weighted_signal_yield_s"], res["weighted_background_yield_b"], br=10.0)
    assert pytest.approx(res["ams_score"], rel=1e-7) == expected_ams
