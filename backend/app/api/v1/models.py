from typing import List

from backend.app.schemas.metrics import ModelMetricsResponse
from backend.app.schemas.models import ModelListResponse, ModelSummarySchema
from backend.app.services.metrics_service import metrics_service
from backend.app.services.model_registry import model_registry_service
from fastapi import APIRouter

router = APIRouter()


@router.get("", response_model=ModelListResponse, tags=["Models"])
def list_models() -> ModelListResponse:
    """Returns registered ML model candidates with headline metrics loaded directly from metrics.json."""
    model_ids = model_registry_service.list_model_ids()
    summaries: List[ModelSummarySchema] = []
    for m_id in model_ids:
        artifact = model_registry_service.get_artifact(m_id)
        m = artifact.metrics
        summaries.append(
            ModelSummarySchema(
                model_id=m_id,
                display_name=m.get("model_id", m_id).replace("_", " ").title(),
                roc_auc=float(m.get("roc_auc_mean", 0.0)),
                ams_score=float(m.get("ams_score", 0.0)),
                optimal_threshold=float(m.get("optimal_threshold", 0.6862)),
                status="available" if artifact.has_weights else "weights_missing",
                weights_available=artifact.has_weights
            )
        )
    return ModelListResponse(models=summaries)


@router.get("/{model_id}/metrics", response_model=ModelMetricsResponse, tags=["Models"])
def get_model_metrics(model_id: str) -> ModelMetricsResponse:
    """Returns full stored evaluation metric set for a specific pre-trained model artifact."""
    return metrics_service.get_full_metrics(model_id)
