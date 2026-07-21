from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReconstructedObject(BaseModel):
    object_type: str  # "hadronic_tau", "lepton", "leading_jet", "subleading_jet"
    label: str
    pt: float
    eta: float
    phi: float
    present: bool


class MissingTransverseEnergy(BaseModel):
    magnitude: float
    phi: float
    sumet: Optional[float] = None


class DetectorViewMetadata(BaseModel):
    coordinate_system: str = "transverse"
    units: Dict[str, str] = {"momentum": "GeV", "angles": "radians"}
    visual_label: str = "Reconstructed event-object visualization"


class PredictionSummary(BaseModel):
    model_id: str
    model_version: str
    feature_set: str
    signal_probability: float
    background_probability: float
    predicted_class: str  # "signal" or "background"
    decision_threshold: float
    distance_from_threshold: float
    validation_status: str
    explanation_status: str = "SHAP / explanation plugin not installed or inactive"


class PredictionResponse(BaseModel):
    event_id: Optional[int] = None
    objects: List[ReconstructedObject]
    missing_transverse_energy: MissingTransverseEnergy
    jet_summary: Dict[str, Any]  # count, total_pt
    detector_view: DetectorViewMetadata
    prediction: PredictionSummary
    missing_adjusted_fields: List[str] = []


class PredictionRequest(BaseModel):
    model_id: str
    feature_set: str = "all_physics"
    event_id: Optional[int] = None
    features: Optional[Dict[str, float]] = None


class ComparePredictionsRequest(BaseModel):
    model_ids: List[str]
    feature_set: str = "all_physics"
    event_id: Optional[int] = None
    features: Optional[Dict[str, float]] = None


class ComparePredictionsResponse(BaseModel):
    event_id: Optional[int] = None
    objects: List[ReconstructedObject]
    missing_transverse_energy: MissingTransverseEnergy
    detector_view: DetectorViewMetadata
    predictions: Dict[str, PredictionSummary]
