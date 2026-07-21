from typing import Any, Dict

import numpy as np
import pandas as pd
from backend.app.data.pipeline import DatasetPipeline
from backend.app.models.base import ModelCandidate
from backend.app.physics.metrics import evaluate_threshold_scan
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


class ModelEvaluator:
    """
    Evaluates candidate models strictly on validation set (`b` partition) to compute
    ROC-AUC, PR-AUC, Log Loss, Balanced Accuracy, F1, Brier score calibration, and optimal AMS threshold.
    Ensures zero test-set leakage.
    """
    @staticmethod
    def evaluate_model_on_val(
        candidate: ModelCandidate,
        df_val: pd.DataFrame,
        feature_set: str = "all_physics",
        ams_br: float = 10.0
    ) -> Dict[str, Any]:
        y_true = DatasetPipeline.encode_labels(df_val)
        y_probs = candidate.predict_proba(df_val)[:, 1]
        weights = df_val["Weight"].to_numpy() if "Weight" in df_val.columns else np.ones_like(y_true, dtype=np.float64)

        # Compute threshold scan on validation data only
        opt_thresh, max_ams, ams_05 = evaluate_threshold_scan(y_true, y_probs, weights, br=ams_br)

        # Predictions at optimal validation threshold
        y_pred = (y_probs >= opt_thresh).astype(np.int32)

        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_probs))
        except Exception:
            pr_auc = 0.0
        try:
            ll = float(log_loss(y_true, y_probs, eps=1e-15))
        except Exception:
            ll = 1.0
        try:
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            bal_acc = 0.5
        try:
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            f1 = 0.0
        try:
            brier = float(brier_score_loss(y_true, y_probs))
        except Exception:
            brier = 0.25

        # Check calibration status
        # If Brier score < 0.18, reasonably calibrated
        calibration_status = "calibrated" if brier < 0.18 else "uncalibrated"

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": ll,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "brier_score": brier,
            "optimal_threshold": opt_thresh,
            "ams_score": max_ams,
            "ams_05_score": ams_05,
            "calibration_status": calibration_status
        }
