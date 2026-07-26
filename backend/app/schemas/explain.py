"""
Pydantic v2 schemas for Feature Explanation & Attribution API (/api/v1/explain).
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ExplainRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary containing all 30 physical feature values"
    )
    model_id: str = Field(
        "xgboost",
        description="Certified tree-booster model ID (defaults to 'xgboost')"
    )

    @field_validator("features")
    @classmethod
    def validate_features_non_empty(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Payload 'features' dictionary cannot be empty.")
        return v


class FeatureAttribution(BaseModel):
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value in event payload")
    contribution: float = Field(..., description="TreeSHAP contribution in log-odds space")


class ObjectGroupAttribution(BaseModel):
    group: str = Field(..., description="Physics object group (tau, lepton, leading_jet, subleading_jet, met, global)")
    total_abs_contribution: float = Field(..., description="Sum of absolute TreeSHAP contributions for group features")
    signed_contribution: float = Field(..., description="Net signed sum of TreeSHAP contributions for group features")


class ExplainResponse(BaseModel):
    model_id: str = Field(..., description="Certified model ID evaluated")
    probability: float = Field(..., description="Signal probability from certified PredictionService")
    predicted_label: str = Field(..., description="Predicted classification label ('signal' or 'background')")
    threshold: float = Field(..., description="Decision threshold applied from model metrics.json")
    base_value: float = Field(..., description="TreeSHAP expected value / bias in log-odds space")
    margin: float = Field(..., description="Total log-odds sum (base_value + sum(contributions))")
    attributions: List[FeatureAttribution] = Field(
        ...,
        description="Per-feature attributions for all 30 features, sorted by |contribution| descending"
    )
    object_groups: List[ObjectGroupAttribution] = Field(
        ...,
        description="Aggregated attributions across 6 canonical physics object groups"
    )
