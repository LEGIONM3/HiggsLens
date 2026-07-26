"""
Pydantic v2 schemas for Event Gallery & Permalink Endpoints (/api/v1/events/gallery & /permalink).
"""

from typing import Dict, List, Literal, Optional

from backend.app.schemas.explain import ExplainResponse
from pydantic import BaseModel, Field

GalleryCategory = Literal["signal", "background", "interesting"]


class GalleryEventSummary(BaseModel):
    event_id: int = Field(..., description="Unique EventId")
    features: Dict[str, float] = Field(..., description="All 30 physical features")
    signal_probability: float = Field(..., description="Signal probability from certified PredictionService")
    predicted_label: str = Field(..., description="Predicted label ('signal' or 'background')")
    threshold: float = Field(..., description="Optimal decision threshold from metrics.json")
    gallery_category: GalleryCategory = Field(..., description="Category: 'signal', 'background', or 'interesting'")
    gallery_rank: int = Field(..., description="1-indexed rank within category")
    selection_method: str = Field("threshold_window", description="Selection method ('threshold_window' or 'nearest_threshold_fallback')")


class GalleryResponse(BaseModel):
    events: List[GalleryEventSummary] = Field(..., description="Curated gallery events")
    total_count: int = Field(..., description="Total events in gallery")
    categories: Dict[str, int] = Field(..., description="Event count per category")
    selection_method: str = Field("threshold_window", description="Global selection method note")


class PermalinkResponse(BaseModel):
    event_id: int = Field(..., description="Unique EventId")
    features: Dict[str, float] = Field(..., description="All 30 physical features")
    kaggleset: str = Field(..., description="Dataset split partition identifier (e.g. 'v' or 't')")
    signal_probability: float = Field(..., description="Signal probability")
    predicted_label: str = Field(..., description="Predicted label ('signal' or 'background')")
    threshold: float = Field(..., description="Optimal threshold")
    explanation: ExplainResponse = Field(..., description="TreeSHAP feature attributions and object groups")
