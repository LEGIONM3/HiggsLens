"""
Pydantic v2 schemas for Event Sampling API (/api/v1/events).
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EventPredictionResponse(BaseModel):
    model_id: str = Field(..., description="ID of the certified model used for inference")
    probability: float = Field(..., description="Predicted signal probability (0.0 to 1.0)")
    predicted_label: str = Field(..., description="Binary prediction label: 'signal' or 'background'")
    threshold: float = Field(..., description="Certified decision threshold read dynamically from artifact")


class EventDataResponse(BaseModel):
    event_id: int = Field(..., description="Unique CERN/ATLAS event ID")
    true_label: str = Field(..., description="True label from ATLAS open data: 'signal' or 'background'")
    features: Dict[str, float] = Field(..., description="All 30 raw physics features (PRI_*/DER_*), sentinels as -999.0")
    prediction: EventPredictionResponse = Field(..., description="Certified champion model prediction payload")


class EventSampleResponse(BaseModel):
    events: List[EventDataResponse] = Field(..., description="List of sampled test-split events")
    count: int = Field(..., description="Number of events returned")
    seed: int = Field(..., description="Random seed used for reproducible sampling")
    label_filter: str = Field(..., description="Label filter applied ('any', 'signal', 'background')")
