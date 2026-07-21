from backend.app.core.config import settings
from fastapi import APIRouter
from pydantic import BaseModel


class ConfigResponse(BaseModel):
    app_name: str = settings.APP_NAME
    version: str = settings.APP_VERSION
    debug: bool = settings.DEBUG
    cern_record_id: int = settings.CERN_RECORD_ID
    cern_metadata_url: str = settings.CERN_METADATA_URL
    canonical_dataset_url: str = settings.CANONICAL_DATASET_URL
    dataset_filename: str = settings.DATASET_FILENAME


router = APIRouter()


@router.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse()
