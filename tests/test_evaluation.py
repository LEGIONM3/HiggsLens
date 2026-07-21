import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from backend.app.evaluation.evaluator import ModelEvaluator
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo


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

    def check_dependency(self) -> bool:
        return True

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id="mock", display_name="Mock", status="available", required=False,
            supported_feature_sets=["all_physics"], supports_missing=False,
            preprocessing_pipeline="native", hyperparameters_schema={}
        )


def test_log_loss_not_one_and_exact_match():
    # Test with known labels and probabilities where expected log loss is NOT 1.0000
    df_val = pd.DataFrame({
        "Label": ["s", "b", "b", "s"],
        "Weight": [1.0, 1.0, 1.0, 1.0],
        "EventId": [1, 2, 3, 4]
    })
    # Probabilities for class 0 and class 1
    probs = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.15, 0.85]
    ])
    cand = MockCandidate(probs=probs)
    res = ModelEvaluator.evaluate_model_on_val(cand, df_val)

    # Calculate expected log loss directly with sklearn
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
    # Out of bounds
    cand = MockCandidate(probs=np.array([[-0.1, 1.1], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="outside \\[0,1\\]"):
        ModelEvaluator.evaluate_model_on_val(cand, df_val)

    # NaN probabilities
    cand = MockCandidate(probs=np.array([[np.nan, 0.5], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="contain NaN or infinity"):
        ModelEvaluator.evaluate_model_on_val(cand, df_val)

    # Do not sum to 1
    cand = MockCandidate(probs=np.array([[0.3, 0.3], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="do not sum approximately to 1"):
        ModelEvaluator.evaluate_model_on_val(cand, df_val)


def test_predict_proba_class_ordering():
    df_val = pd.DataFrame({
        "Label": ["s", "b"],
        "Weight": [1.0, 1.0],
        "EventId": [1, 2]
    })
    # If classes_ is [1, 0] instead of [0, 1], signal prob is column 0
    probs = np.array([
        [0.9, 0.1],  # row 1: prob of class 1 is 0.9, prob of class 0 is 0.1
        [0.2, 0.8]   # row 2: prob of class 1 is 0.2, prob of class 0 is 0.8
    ])
    cand = MockCandidate(probs=probs, classes=np.array([1, 0]))
    res = ModelEvaluator.evaluate_model_on_val(cand, df_val)
    assert res["roc_auc"] == 1.0


def test_ams_invariant_and_formula_agreement():
    from backend.app.physics.metrics import compute_ams
    # Verify reported AMS exactly equals direct compute_ams(reported_s, reported_b, br)
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


def test_positive_finite_s_nonnegative_b_not_silent_zero():
    from backend.app.physics.metrics import compute_ams
    # Positive finite s with nonnegative b cannot silently produce zero unless formula yields zero
    s = 84.25
    b = 50463.91
    br = 10.0
    ams_val = compute_ams(s, b, br)
    assert ams_val > 0.0, "Positive signal yield with valid background must produce non-zero AMS"
    assert pytest.approx(ams_val, rel=1e-5) == 0.37490


def test_dummy_classifier_threshold_behavior():
    from backend.app.models.dummy import DummyPriorCandidate
    from backend.app.physics.metrics import compute_ams
    df_train = pd.DataFrame({
        "PRI_tau_pt": [10.0, 20.0, 30.0],
        "Label": ["s", "b", "b"]
    })
    df_val = pd.DataFrame({
        "PRI_tau_pt": [15.0, 25.0, 35.0, 45.0],
        "Label": ["s", "s", "b", "b"],
        "Weight": [10.0, 20.0, 100.0, 200.0],
        "EventId": [1, 2, 3, 4]
    })
    y_train = np.array([1, 0, 0])
    cand = DummyPriorCandidate().fit(df_train, y_train, feature_set="primary_only")
    res = ModelEvaluator.evaluate_model_on_val(cand, df_val, ams_br=10.0)
    # Since prior is 1/3 ~ 0.333, and evaluate_threshold_scan checks thresholds from 0.01 to 0.99,
    # at threshold 0.01 (<= constant prob), every validation event is classified as signal.
    total_s = df_val.loc[df_val["Label"] == "s", "Weight"].sum()
    total_b = df_val.loc[df_val["Label"] == "b", "Weight"].sum()
    assert res["weighted_signal_yield_s"] == total_s
    assert res["weighted_background_yield_b"] == total_b
    expected_ams = compute_ams(total_s, total_b, br=10.0)
    assert pytest.approx(res["ams_score"], rel=1e-7) == expected_ams

