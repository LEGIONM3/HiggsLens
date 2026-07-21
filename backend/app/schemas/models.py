from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelHyperparameterSchema(BaseModel):
    name: str
    param_type: str  # int, float, str, bool, list
    default_value: Any
    description: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    display_name: str
    status: str  # "available", "unavailable", "incompatible"
    required: bool
    supported_feature_sets: List[str] = ["primary_only", "derived_only", "all_physics"]
    supports_missing: bool
    preprocessing_pipeline: str
    hyperparameters_schema: Dict[str, Any]
    probability_support: bool = True


class ModelMetricsSchema(BaseModel):
    model_id: str
    feature_set: str
    mode: str  # "fast" or "research"
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
    stability_status: str  # "stable", "high_variance", "overfitting_warning"
    calibration_status: str  # "calibrated", "uncalibrated"
    model_card_path: Optional[str] = None


class ModelRegistryResponse(BaseModel):
    models: Dict[str, ModelInfo]
