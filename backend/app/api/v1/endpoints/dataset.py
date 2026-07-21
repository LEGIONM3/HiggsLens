import json
from datetime import datetime, timezone
from typing import Optional

from backend.app.core.config import settings
from backend.app.data.downloader import CERNDataDownloader
from backend.app.data.validator import DatasetValidator
from backend.app.schemas.dataset import (
    DatasetSchemaResponse,
    DatasetStatusResponse,
    DatasetValidationReport,
)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


@router.get("/status", response_model=DatasetStatusResponse)
def get_dataset_status():
    raw_path = settings.get_raw_dataset_path()
    exists = raw_path.exists()
    size = raw_path.stat().st_size if exists else 0
    last_mod = datetime.fromtimestamp(raw_path.stat().st_mtime, timezone.utc).isoformat() if exists else None

    # Check if a validation report already exists in artifacts
    report: Optional[DatasetValidationReport] = None
    report_file = settings.ARTIFACTS_DIR / "metrics" / "dataset_validation_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r") as f:
                data = json.load(f)
                report = DatasetValidationReport.model_validate(data)
        except Exception:
            report = None

    return DatasetStatusResponse(
        exists=exists,
        filepath=str(raw_path),
        file_size_bytes=size,
        record_id=settings.CERN_RECORD_ID,
        doi="10.7483/OPENDATA.ATLAS.ZBP2.M5T8",
        last_modified=last_mod,
        validation_report=report
    )


class DownloadRequest(BaseModel):
    force: bool = False
    dry_run: bool = False


class DownloadResponse(BaseModel):
    success: bool
    message: str


@router.post("/download", response_model=DownloadResponse)
def download_dataset(req: DownloadRequest):
    downloader = CERNDataDownloader()
    success, msg = downloader.download(force=req.force, dry_run=req.dry_run)
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return DownloadResponse(success=success, message=msg)


@router.post("/validate", response_model=DatasetValidationReport)
def validate_dataset():
    raw_path = settings.get_raw_dataset_path()
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Download it via POST /api/v1/dataset/download first.")
    try:
        validator = DatasetValidator(raw_path)
        return validator.validate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.get("/schema", response_model=DatasetSchemaResponse)
def get_dataset_schema():
    return DatasetSchemaResponse()


@router.get("/summary")
def get_dataset_summary():
    report_file = settings.ARTIFACTS_DIR / "metrics" / "dataset_validation_report.json"
    if not report_file.exists():
        # Check if dataset exists to validate on the fly
        raw_path = settings.get_raw_dataset_path()
        if not raw_path.exists():
            return {"status": "missing", "message": "Dataset not downloaded yet."}
        validator = DatasetValidator(raw_path)
        report = validator.validate()
        return report.model_dump()
    try:
        with open(report_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
