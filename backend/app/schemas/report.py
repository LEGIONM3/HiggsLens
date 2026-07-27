"""
Pydantic v2 schemas for HiggsLens Research Reports and Reproducibility Manifests.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.explain import FeatureAttribution, ObjectGroupAttribution


class ReportGalleryCategory(BaseModel):
    category: Optional[str] = Field(None, description="Gallery category ('signal', 'background', 'interesting') or None")
    rank: Optional[int] = Field(None, description="Gallery ranking if present")
    selection_method: Optional[str] = Field(None, description="Selection method used in gallery")


class ReportEventSummary(BaseModel):
    event_id: int
    features: Dict[str, float]
    source_split: str = "test"
    gallery: Optional[ReportGalleryCategory] = None


class ReportClassification(BaseModel):
    model_id: Literal["xgboost"] = "xgboost"
    signal_probability: float
    predicted_label: Literal["signal", "background"]
    threshold: float


class ReportExplanation(BaseModel):
    base_value: float
    margin: float
    attributions: List[FeatureAttribution]
    object_groups: List[ObjectGroupAttribution]


class ReproducibilityModelArtifact(BaseModel):
    model_id: str = "xgboost"
    feature_schema_version: str = "v1"
    device: str = "CPU (scikit-learn)"
    training_run_origin: str = "Official baseline"
    subsample_notes: str = "Full dataset"


class ReproducibilityDataset(BaseModel):
    record: str = "328"
    doi: str = "10.7483/OPENDATA.ATLAS.ZBP2.M5T8"
    content_hash: str = "54242acf28a78ce303ea48bcf7002f0a44df08448271477e0a63331486c4f316"


class ReproducibilityInferenceContract(BaseModel):
    feature_count: int = 30
    sentinel_value: float = -999.0
    prediction_path: str = "certified PredictionService"
    explanation_path: str = "native XGBoost pred_contribs=True"


class ReproducibilityManifest(BaseModel):
    report_contract_version: str = "1.0"
    dataset: ReproducibilityDataset
    inference_contract: ReproducibilityInferenceContract
    certified_models: List[ReproducibilityModelArtifact]
    frozen_leaderboard_status: str = "Certified & Frozen"
    server_capabilities: Dict[str, Any] = Field(
        default_factory=lambda: {
            "supported_report_formats": ["json", "html"],
            "retraining_supported": False,
            "public_holdout_access": False,
            "champion_model_id": "xgboost"
        }
    )


class EventAnalysisReport(BaseModel):
    report_version: str = "1.0"
    generated_at: str = Field(..., description="Report generation time in ISO-8601 UTC format")
    event: ReportEventSummary
    classification: ReportClassification
    explanation: ReportExplanation
    reproducibility: ReproducibilityManifest
    provenance: Dict[str, str] = Field(
        default_factory=lambda: {
            "statement": "ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8) — official ATLAS simulated events, classified by certified pre-trained models."
        }
    )
