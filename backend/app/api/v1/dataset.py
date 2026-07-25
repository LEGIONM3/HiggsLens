from backend.app.schemas.dataset import DatasetSummaryResponse
from backend.app.services.dataset_service import dataset_service
from fastapi import APIRouter

router = APIRouter()


@router.get("/summary", response_model=DatasetSummaryResponse, tags=["Dataset"])
def get_dataset_summary() -> DatasetSummaryResponse:
    """Returns official dataset facts for CERN Open Data Record 328 (818,238 events, 30 features)."""
    return dataset_service.get_summary()
