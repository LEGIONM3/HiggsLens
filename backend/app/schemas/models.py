from typing import Any, Dict, List, Optional

from backend.app.schemas.metrics import ModelMetricsResponse
from pydantic import BaseModel, ConfigDict

ModelMetricsSchema = ModelMetricsResponse


class ModelInfo(BaseModel):
    id: str
    display_name: str
    status: str
    required: bool
    supported_feature_sets: List[str] = ["all_physics", "primary_only", "derived_only"]
    supports_missing: bool
    preprocessing_pipeline: str
    hyperparameters_schema: Dict[str, Any] = {}

    model_config = ConfigDict(extra="ignore")


class ModelSummarySchema(BaseModel):
    model_id: str
    display_name: str
    roc_auc: float
    ams_score: float
    optimal_threshold: float
    status: str
    weights_available: bool

    model_config = ConfigDict(frozen=True)


class ModelListResponse(BaseModel):
    models: List[ModelSummarySchema]

    model_config = ConfigDict(frozen=True)


class HealthResponse(BaseModel):
    status: str
    version: str
    available_models_count: int

    model_config = ConfigDict(frozen=True)
