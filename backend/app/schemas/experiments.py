from typing import Dict, List, Optional

from pydantic import BaseModel

from .models import ModelMetricsSchema


class TrainingRequest(BaseModel):
    mode: str = "fast"  # "fast" or "research"
    feature_set: str = "all_physics"
    models: Optional[List[str]] = None  # None = train all available
    seeds: Optional[List[int]] = None


class JobStatusResponse(BaseModel):
    job_id: str
    state: str  # queued, validating_data, preprocessing, training, evaluating, saving_artifacts, completed, failed
    current_model: Optional[str] = None
    completed_models: int = 0
    total_models: int = 0
    progress_message: str
    started_timestamp: str
    updated_timestamp: str
    error_details: Optional[str] = None
    run_id: Optional[str] = None


class RecommendationExplanation(BaseModel):
    recommended_model_id: str
    validation_objective_used: str
    feature_set_used: str
    mean_roc_auc: float
    variance_across_seeds: float
    calibration_quality: str
    training_duration_seconds: float
    selected_threshold: float
    weighted_ams: float
    warnings: List[str]
    reasoning: str


class ExperimentRunResponse(BaseModel):
    run_id: str
    timestamp: str
    git_commit: Optional[str]
    dataset_fingerprint: str
    split_strategy: str
    feature_set: str
    mode: str
    seeds: List[int]
    metrics_by_model: Dict[str, ModelMetricsSchema]
    recommendation: Optional[RecommendationExplanation] = None
