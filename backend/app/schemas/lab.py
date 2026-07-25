"""
Pydantic v2 Schemas for HiggsLens Lab Sandboxed Zone.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LabDatasetManifestSchema(BaseModel):
    """Manifest describing an uploaded custom lab dataset."""

    dataset_id: str
    filename: str
    row_count: int
    column_count: int
    feature_columns: List[str]
    label_column: str
    weight_column: Optional[str] = None
    created_at: str
    content_hash: str


class LabDatasetListResponse(BaseModel):
    """Response schema for listing uploaded lab datasets."""

    datasets: List[LabDatasetManifestSchema]


class LabExperimentCreateRequest(BaseModel):
    """Request schema for initiating a custom model training experiment."""

    dataset_id: str
    model_ids: List[str] = Field(..., min_length=1, max_length=5)
    split_config: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.70, "validation": 0.15, "test": 0.15}
    )
    seed: int = 42
    sentinel_strategy: str = "keep-as-value"


class LabExperimentSummarySchema(BaseModel):
    """Summary schema for a lab experiment training job."""

    experiment_id: str
    dataset_id: str
    status: str  # "queued", "running", "completed", "failed"
    model_ids: List[str]
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class LabExperimentListResponse(BaseModel):
    """Response schema for listing lab experiments."""

    experiments: List[LabExperimentSummarySchema]


class LabExperimentDetailResponse(BaseModel):
    """Detailed response schema for a completed or running experiment leaderboard."""

    summary: LabExperimentSummarySchema
    dataset_manifest: Optional[LabDatasetManifestSchema] = None
    split_config: Dict[str, float]
    seed: int
    per_model_results: Dict[str, Any] = Field(default_factory=dict)
