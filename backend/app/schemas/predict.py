from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="ID of pre-trained model artifact to execute inference with.")
    features: Dict[str, float] = Field(..., description="Feature vector mapping 30 feature names to float values.")
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Decision threshold for signal classification (defaults to model's optimal threshold or 0.6862)."
    )


class PredictResponse(BaseModel):
    signal_probability: float = Field(..., description="Predicted probability of signal topology.")
    predicted_label: int = Field(..., description="Predicted label (1 for signal, 0 for background).")
    threshold_used: float = Field(..., description="Decision threshold applied for binary classification.")
    model_id: str = Field(..., description="Model ID used for inference.")
    manifest: Dict[str, Any] = Field(..., description="Provenance manifest reference (training commit, seed, dataset hash).")
