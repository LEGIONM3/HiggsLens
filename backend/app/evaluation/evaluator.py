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
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
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
        y_probs_2d = candidate.predict_proba(df_val)

        # Quality Gates: Fail if probabilities invalid or unnormalized
        if np.isnan(y_probs_2d).any() or np.isinf(y_probs_2d).any():
            raise ValueError("Probabilities contain NaN or infinity")
        if (y_probs_2d < 0.0).any() or (y_probs_2d > 1.0).any():
            raise ValueError("A probability is outside [0,1]")
        if y_probs_2d.shape[1] >= 2 and not np.allclose(y_probs_2d.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Signal and background probabilities do not sum approximately to 1")

        # Verify class ordering
        if hasattr(candidate.model, "classes_") and candidate.model.classes_ is not None:
            sig_idx = np.where(candidate.model.classes_ == 1)[0]
            sig_col = sig_idx[0] if len(sig_idx) > 0 else (1 if y_probs_2d.shape[1] > 1 else 0)
        else:
            sig_col = 1 if y_probs_2d.shape[1] > 1 else 0
        y_probs = y_probs_2d[:, sig_col]

        weights = df_val["Weight"].to_numpy() if "Weight" in df_val.columns else np.ones_like(y_true, dtype=np.float64)

        # Compute threshold scan on validation data only
        opt_thresh, max_ams, ams_05 = evaluate_threshold_scan(y_true, y_probs, weights, br=ams_br)

        # Predictions at optimal validation threshold and default 0.5 threshold
        y_pred_opt = (y_probs >= opt_thresh).astype(np.int32)
        y_pred_05 = (y_probs >= 0.5).astype(np.int32)

        # Exact log_loss with clipped probabilities (no fallback constant allowed)
        y_probs_clipped = np.clip(y_probs, 1e-15, 1.0 - 1e-15)
        ll = float(log_loss(y_true, y_probs_clipped))

        roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
        bal_acc = float(balanced_accuracy_score(y_true, y_pred_opt))
        f1 = float(f1_score(y_true, y_pred_opt, zero_division=0))
        brier = float(brier_score_loss(y_true, y_probs))

        # Yields at selected threshold
        mask_selected = y_probs >= opt_thresh
        s_yield = float(np.sum(weights[(y_true == 1) & mask_selected]))
        b_yield = float(np.sum(weights[(y_true == 0) & mask_selected]))

        # Confusion matrices & precision/recall
        cm_05 = confusion_matrix(y_true, y_pred_05, labels=[0, 1])
        tn_05, fp_05, fn_05, tp_05 = int(cm_05[0, 0]), int(cm_05[0, 1]), int(cm_05[1, 0]), int(cm_05[1, 1])
        prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
        rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))

        cm_opt = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])
        tn_opt, fp_opt, fn_opt, tp_opt = int(cm_opt[0, 0]), int(cm_opt[0, 1]), int(cm_opt[1, 0]), int(cm_opt[1, 1])
        prec_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
        rec_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))

        # Reliability bins & Expected Calibration Error
        bins = np.linspace(0.0, 1.0, 11)
        bin_indices = np.digitize(y_probs, bins) - 1
        reliability_bins = []
        ece = 0.0
        n_total = len(y_probs)
        for i in range(10):
            mask = bin_indices == i
            if np.any(mask):
                bin_count = int(np.sum(mask))
                bin_conf = float(np.mean(y_probs[mask]))
                bin_acc = float(np.mean(y_true[mask]))
                reliability_bins.append({
                    "bin_low": float(bins[i]),
                    "bin_high": float(bins[i+1]),
                    "confidence": bin_conf,
                    "accuracy": bin_acc,
                    "count": bin_count
                })
                ece += (bin_count / max(1, n_total)) * abs(bin_acc - bin_conf)

        return {
            "validation_rows": int(len(df_val)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": ll,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "brier_score": brier,
            "optimal_threshold": opt_thresh,
            "ams_score": max_ams,
            "ams_05_score": ams_05,
            "precision_05": prec_05,
            "recall_05": rec_05,
            "precision_selected": prec_opt,
            "recall_selected": rec_opt,
            "confusion_matrix_05": {"tn": tn_05, "fp": fp_05, "fn": fn_05, "tp": tp_05},
            "confusion_matrix_selected": {"tn": tn_opt, "fp": fp_opt, "fn": fn_opt, "tp": tp_opt},
            "weighted_signal_yield_s": s_yield,
            "weighted_background_yield_b": b_yield,
            "ams_br": float(ams_br),
            "expected_calibration_error": float(ece),
            "reliability_bins": reliability_bins
        }
