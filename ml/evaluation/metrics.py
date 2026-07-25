"""
Physics Metric Calculation Utilities & Threshold Sweeper for HiggsLens.
Includes AMS calculation and Luminosity Weight Renormalization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_ams(s: float, b: float, br: float = 10.0) -> float:
    """
    Computes Approximate Median Significance (AMS) for collider discovery sensitivity.

    Formula:
      AMS = sqrt(2 * ((s + b + br) * ln(1 + s / (b + br)) - s))

    Returns 0.0 if s <= 0 or b + br <= 0.
    """
    if s <= 0.0 or (b + br) <= 0.0:
        return 0.0

    radicand = 2.0 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    if radicand <= 0.0:
        return 0.0

    return float(np.sqrt(radicand))


def evaluate_threshold_scan(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    weights: np.ndarray,
    br: float = 10.0,
    target_luminosity_scale: Optional[Dict[str, float]] = None,
    num_thresholds: int = 100
) -> Tuple[float, float, float, List[Dict[str, float]]]:
    """
    Scans decision thresholds from 0.01 to 0.99 to find optimal AMS threshold.
    If target_luminosity_scale is provided ({'full_signal_weight': S_full, 'full_background_weight': B_full}),
    renormalizes split weights to full-dataset target luminosity.

    Returns:
      (optimal_threshold, max_ams_score, ams_at_default_threshold_05, threshold_curve_points)
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Luminosity weight renormalization
    w_effective = weights.copy()
    if target_luminosity_scale:
        s_split = float(weights[y_true == 1].sum())
        b_split = float(weights[y_true == 0].sum())

        s_target = target_luminosity_scale.get("full_signal_weight", s_split)
        b_target = target_luminosity_scale.get("full_background_weight", b_split)

        f_s = (s_target / s_split) if s_split > 0 else 1.0
        f_b = (b_target / b_split) if b_split > 0 else 1.0

        w_effective[y_true == 1] *= f_s
        w_effective[y_true == 0] *= f_b

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_t = 0.5
    max_ams = 0.0
    ams_05 = 0.0
    curve_points: List[Dict[str, float]] = []

    for t in thresholds:
        selected = y_probs >= t
        s_yield = float(w_effective[(y_true == 1) & selected].sum())
        b_yield = float(w_effective[(y_true == 0) & selected].sum())

        ams_val = compute_ams(s_yield, b_yield, br=br)

        # Calculate unweighted precision, recall, f1 for curve metadata
        tp = int(((y_true == 1) & selected).sum())
        fp = int(((y_true == 0) & selected).sum())
        fn = int(((y_true == 1) & ~selected).sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        curve_points.append({
            "threshold": float(t),
            "ams": ams_val,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "signal_yield": s_yield,
            "background_yield": b_yield,
        })

        if ams_val > max_ams:
            max_ams = ams_val
            best_t = float(t)

        if abs(t - 0.5) < 0.01:
            ams_05 = ams_val

    return best_t, max_ams, ams_05, curve_points
