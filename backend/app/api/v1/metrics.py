from backend.app.schemas.metrics import ThresholdCurveResponse
from backend.app.services.metrics_service import metrics_service
from fastapi import APIRouter

router = APIRouter()


@router.get("/{model_id}/thresholds", response_model=ThresholdCurveResponse, tags=["Metrics"])
def get_threshold_curve(model_id: str) -> ThresholdCurveResponse:
    """Returns stored AMS/threshold curve data points for frontend charting."""
    return metrics_service.get_threshold_curve(model_id)
