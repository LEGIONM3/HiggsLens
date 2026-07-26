"""
Pydantic v2 schemas for Feature Derivation API (/api/v1/events/derive).
"""

from typing import Dict, List, Optional

from backend.app.schemas.events import EventPredictionResponse
from pydantic import BaseModel, Field, field_validator


class DeriveRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary containing all 17 raw primary (PRI_*) feature values"
    )
    base_event_id: Optional[int] = Field(
        None,
        description="Optional base EventId to preserve original stored DER_mass_MMC if PRI features match"
    )
    model_id: str = Field(
        "xgboost",
        description="Certified model ID for inference prediction (defaults to 'xgboost')"
    )

    @field_validator("features")
    @classmethod
    def validate_features_non_empty(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Payload 'features' dictionary cannot be empty.")
        return v


class DeriveResponse(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Full 30-feature dictionary (17 PRI_* + 13 DER_*) after re-derivation"
    )
    prediction: EventPredictionResponse = Field(
        ...,
        description="Certified model prediction response for the assembled 30-feature vector"
    )
    mmc_policy: str = Field(
        ...,
        description="MMC policy applied: 'original' if matched base_event_id, or 'sentinel' (-999.0)"
    )
    notes: List[str] = Field(
        ...,
        description="Explanatory notes regarding DER calculations, MMC policy, or sentinels"
    )
