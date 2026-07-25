from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ModelMetricsResponse(BaseModel):
    model_id: str
    feature_set: str
    mode: str
    seeds_evaluated: List[int]
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    log_loss_mean: float
    log_loss_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float
    f1_mean: float
    f1_std: float
    brier_score_mean: float
    brier_score_std: float
    optimal_threshold: float
    ams_score: float
    ams_default_threshold_score: float
    training_duration_seconds: float
    stability_status: str
    calibration_status: str
    validation_rows: int
    precision_05: float
    recall_05: float
    precision_selected: float
    recall_selected: float
    confusion_matrix_05: Dict[str, int]
    confusion_matrix_selected: Dict[str, int]
    weighted_signal_yield_s: float
    weighted_background_yield_b: float
    ams_br: float
    calibration_method: str
    expected_calibration_error: float
    reliability_bins: List[Dict[str, Any]]

    model_config = ConfigDict(extra="ignore")


class ThresholdPoint(BaseModel):
    threshold: float = Field(..., description="Classification probability threshold")
    ams: float = Field(..., description="Approximate Median Significance score")
    precision: float = Field(..., description="Precision at threshold")
    recall: float = Field(..., description="Recall at threshold")
    f1: float = Field(..., description="F1 score at threshold")


class ThresholdCurveResponse(BaseModel):
    model_id: str
    optimal_threshold: float
    points: List[ThresholdPoint]

    model_config = ConfigDict(frozen=True)
