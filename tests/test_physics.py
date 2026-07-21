import numpy as np
import pytest

from backend.app.physics.metrics import compute_ams, evaluate_threshold_scan


def test_compute_ams_hand_calculation():
    # s=50, b=20, br=10 -> sqrt(2 * (80 * ln(1 + 50/30) - 50)) ~ 7.545374774
    ams = compute_ams(50.0, 20.0, br=10.0)
    expected = np.sqrt(2.0 * ((50.0 + 20.0 + 10.0) * np.log(1.0 + 50.0 / 30.0) - 50.0))
    assert pytest.approx(ams, rel=1e-7) == expected
    assert pytest.approx(ams, rel=1e-5) == 7.54537


def test_compute_ams_zero_or_negative_signal():
    assert compute_ams(0.0, 10.0) == 0.0
    assert compute_ams(-5.0, 10.0) == 0.0


def test_compute_ams_zero_background():
    # s=10, b=0, br=10 -> sqrt(2 * (20 * ln(2) - 10))
    ams = compute_ams(10.0, 0.0, br=10.0)
    expected = np.sqrt(2.0 * (20.0 * np.log(2.0) - 10.0))
    assert pytest.approx(ams, rel=1e-7) == expected


def test_evaluate_threshold_scan():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_probs = np.array([0.9, 0.8, 0.7, 0.2, 0.85, 0.1])
    weights = np.array([10.0, 10.0, 5.0, 5.0, 10.0, 5.0])

    best_t, max_ams, ams_05 = evaluate_threshold_scan(y_true, y_probs, weights, br=10.0)
    assert 0.0 < best_t < 1.0
    assert max_ams >= ams_05
    assert max_ams > 0.0
