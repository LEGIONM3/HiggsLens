from backend.app.schemas.predict import PredictRequest, PredictResponse
from backend.app.services.prediction_service import prediction_service
from fastapi import APIRouter

router = APIRouter()


@router.post("", response_model=PredictResponse, tags=["Predictions"])
def predict_event(request: PredictRequest) -> PredictResponse:
    """Performs statistical Higgs-event classification inference on validated input feature vector."""
    return prediction_service.predict(request)
