from backend.app.core.config import settings
from backend.app.schemas.models import HealthResponse
from backend.app.services.model_registry import model_registry_service
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Returns system status, application version, and count of available pre-trained model artifacts."""
    model_ids = model_registry_service.list_model_ids()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        available_models_count=len(model_ids)
    )
