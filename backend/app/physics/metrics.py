from typing import Tuple

import numpy as np


def compute_ams(s: float, b: float, br: float = 10.0) -> float:
    """
    Compute Approximate Median Significance (AMS) exactly from the official ATLAS Higgs Challenge 2014 documentation:
    AMS = sqrt(2 * ((s + b + br) * ln(1 + s / (b + br)) - s))
    
    Args:
        s: sum of event weights for true signal events classified as positive
        b: sum of event weights for true background events classified as positive
        br: regularizing term (default 10.0)
    """
    if s <= 0.0:
        return 0.0
    if b < 0.0:
        b = 0.0

    term1 = (s + b + br) * np.log(1.0 + s / (b + br))
    inside_sqrt = 2.0 * (term1 - s)
    if inside_sqrt <= 0.0:
        return 0.0
    return float(np.sqrt(inside_sqrt))


def evaluate_threshold_scan(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    weights: np.ndarray,
    br: float = 10.0,
    num_thresholds: int = 101
) -> Tuple[float, float, float]:
    """
    Scan thresholds across [0.0, 1.0] on validation data to find the threshold yielding maximum AMS.
    Returns: (optimal_threshold, max_ams, ams_at_0_5)
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_threshold = 0.5
    best_ams = 0.0

    # Precalculate AMS at 0.5
    pred_05 = (y_probs >= 0.5)
    s_05 = float(np.sum(weights[(y_true == 1) & pred_05]))
    b_05 = float(np.sum(weights[(y_true == 0) & pred_05]))
    ams_05 = compute_ams(s_05, b_05, br=br)

    for t in thresholds:
        pred = (y_probs >= t)
        s = float(np.sum(weights[(y_true == 1) & pred]))
        b = float(np.sum(weights[(y_true == 0) & pred]))
        ams = compute_ams(s, b, br=br)
        if ams > best_ams:
            best_ams = ams
            best_threshold = float(t)

    return best_threshold, best_ams, ams_05
