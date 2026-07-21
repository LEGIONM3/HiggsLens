from backend.app.core.config import settings
from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str = settings.APP_NAME
    version: str = settings.APP_VERSION


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health():
    return HealthResponse()
