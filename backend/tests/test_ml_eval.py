import numpy as np
import pytest
from ml.evaluation.contract import EvaluationResult
from ml.evaluation.metrics import compute_ams, evaluate_threshold_scan


def test_compute_ams_hand_calculation():
    # s=50, b=20, br=10 -> sqrt(2 * (80 * ln(1 + 50/30) - 50)) ~ 7.54537
    ams = compute_ams(50.0, 20.0, br=10.0)
    expected = np.sqrt(2.0 * ((50.0 + 20.0 + 10.0) * np.log(1.0 + 50.0 / 30.0) - 50.0))
    assert pytest.approx(ams, rel=1e-7) == expected
    assert pytest.approx(ams, rel=1e-5) == 7.54537


def test_compute_ams_edge_cases():
    assert compute_ams(0.0, 10.0) == 0.0
    assert compute_ams(-10.0, 10.0) == 0.0
    assert compute_ams(50.0, -15.0, br=10.0) == 0.0


def test_evaluate_threshold_scan_renormalization():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_probs = np.array([0.9, 0.8, 0.7, 0.2, 0.85, 0.1])
    weights = np.array([10.0, 10.0, 5.0, 5.0, 10.0, 5.0])

    target_lum = {"full_signal_weight": 300.0, "full_background_weight": 150.0}

    best_t, max_ams, ams_05, curve = evaluate_threshold_scan(
        y_true, y_probs, weights, br=10.0, target_luminosity_scale=target_lum
    )

    assert 0.01 <= best_t <= 0.99
    assert max_ams > 0.0
    assert len(curve) == 100
    assert "threshold" in curve[0]
    assert "ams" in curve[0]


def test_evaluation_result_contract_structure():
    res = EvaluationResult(
        model_id="random_forest",
        roc_auc_mean=0.8851,
        ams_score=1.0511,
        optimal_threshold=0.6862
    )
    d = res.to_dict()
    assert d["model_id"] == "random_forest"
    assert d["roc_auc_mean"] == 0.8851
    assert d["ams_score"] == 1.0511
    assert d["optimal_threshold"] == 0.6862
