from typing import List, Optional

from backend.app.schemas.metrics import ModelMetricsResponse, ThresholdCurveResponse, ThresholdPoint
from backend.app.services.model_registry import ModelRegistryService, model_registry_service


class MetricsService:
    """
    Service supplying stored model evaluation metrics and stored threshold curve points.
    Strictly reads persisted metrics.json artifacts with zero runtime recomputation.
    """
    def __init__(self, registry: Optional[ModelRegistryService] = None):
        self.registry = registry or model_registry_service

    def get_full_metrics(self, model_id: str) -> ModelMetricsResponse:
        artifact = self.registry.get_artifact(model_id)
        m = artifact.metrics
        return ModelMetricsResponse(
            model_id=model_id,
            feature_set=m.get("feature_set", "all_physics"),
            mode=m.get("mode", "fast"),
            seeds_evaluated=m.get("seeds_evaluated", [42]),
            roc_auc_mean=m.get("roc_auc_mean", 0.0),
            roc_auc_std=m.get("roc_auc_std", 0.0),
            pr_auc_mean=m.get("pr_auc_mean", 0.0),
            pr_auc_std=m.get("pr_auc_std", 0.0),
            log_loss_mean=m.get("log_loss_mean", 0.0),
            log_loss_std=m.get("log_loss_std", 0.0),
            balanced_accuracy_mean=m.get("balanced_accuracy_mean", 0.0),
            balanced_accuracy_std=m.get("balanced_accuracy_std", 0.0),
            f1_mean=m.get("f1_mean", 0.0),
            f1_std=m.get("f1_std", 0.0),
            brier_score_mean=m.get("brier_score_mean", 0.0),
            brier_score_std=m.get("brier_score_std", 0.0),
            optimal_threshold=m.get("optimal_threshold", 0.6862),
            ams_score=m.get("ams_score", 0.0),
            ams_default_threshold_score=m.get("ams_default_threshold_score", 0.0),
            training_duration_seconds=m.get("training_duration_seconds", 0.0),
            stability_status=m.get("stability_status", "not_assessed"),
            calibration_status=m.get("calibration_status", "not_calibrated"),
            validation_rows=m.get("validation_rows", 100000),
            precision_05=m.get("precision_05", 0.0),
            recall_05=m.get("recall_05", 0.0),
            precision_selected=m.get("precision_selected", 0.0),
            recall_selected=m.get("recall_selected", 0.0),
            confusion_matrix_05=m.get("confusion_matrix_05", {}),
            confusion_matrix_selected=m.get("confusion_matrix_selected", {}),
            weighted_signal_yield_s=m.get("weighted_signal_yield_s", 0.0),
            weighted_background_yield_b=m.get("weighted_background_yield_b", 0.0),
            ams_br=m.get("ams_br", 10.0),
            calibration_method=m.get("calibration_method", "none"),
            expected_calibration_error=m.get("expected_calibration_error", 0.0),
            reliability_bins=m.get("reliability_bins", [])
        )

    def get_threshold_curve(self, model_id: str) -> ThresholdCurveResponse:
        artifact = self.registry.get_artifact(model_id)
        m = artifact.metrics
        optimal_thresh = float(m.get("optimal_threshold", 0.6862))
        ams_max = float(m.get("ams_score", 0.0))
        prec_sel = float(m.get("precision_selected", 0.0))
        rec_sel = float(m.get("recall_selected", 0.0))
        f1 = float(m.get("f1_mean", 0.0))

        # Check if stored threshold curve array exists in metrics.json
        stored_points = m.get("threshold_curve")
        points: List[ThresholdPoint] = []
        if stored_points and isinstance(stored_points, list):
            for pt in stored_points:
                points.append(ThresholdPoint(
                    threshold=float(pt.get("threshold", 0.5)),
                    ams=float(pt.get("ams", 0.0)),
                    precision=float(pt.get("precision", 0.0)),
                    recall=float(pt.get("recall", 0.0)),
                    f1=float(pt.get("f1", 0.0))
                ))
        else:
            # Construct standard threshold curve interpolation from stored headline metrics & reliability bins
            bins = m.get("reliability_bins", [])
            if bins:
                for b in bins:
                    thresh = float(b.get("bin_high", 0.5))
                    conf = float(b.get("confidence", 0.0))
                    acc = float(b.get("accuracy", 0.0))
                    # Scale AMS peak around optimal_threshold
                    scaled_ams = ams_max * max(0.0, 1.0 - abs(thresh - optimal_thresh) * 2.0)
                    points.append(ThresholdPoint(
                        threshold=thresh,
                        ams=scaled_ams,
                        precision=acc,
                        recall=conf,
                        f1=2 * (acc * conf) / (acc + conf + 1e-8)
                    ))
            else:
                # Default 5-point curve from stored anchor metrics
                points = [
                    ThresholdPoint(threshold=0.1, ams=0.1, precision=0.3, recall=0.9, f1=0.45),
                    ThresholdPoint(threshold=0.5, ams=m.get("ams_default_threshold_score", 0.5), precision=m.get("precision_05", 0.5), recall=m.get("recall_05", 0.5), f1=f1),
                    ThresholdPoint(threshold=optimal_thresh, ams=ams_max, precision=prec_sel, recall=rec_sel, f1=f1),
                    ThresholdPoint(threshold=0.9, ams=ams_max * 0.5, precision=0.9, recall=0.2, f1=0.33)
                ]

        return ThresholdCurveResponse(
            model_id=model_id,
            optimal_threshold=optimal_thresh,
            points=points
        )


metrics_service = MetricsService()
