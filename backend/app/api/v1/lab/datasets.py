"""
Lab Dataset Management API Router.
"""

import json
from typing import Optional

from backend.app.schemas.lab import LabDatasetListResponse, LabDatasetManifestSchema
from backend.app.services.lab.dataset_service import lab_dataset_service
from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter(prefix="/lab/datasets", tags=["Lab Datasets"])


@router.post(
    "",
    response_model=LabDatasetManifestSchema,
    summary="Upload custom dataset for sandboxed lab experimentation"
)
async def upload_lab_dataset(
    file: UploadFile = File(...),
    feature_columns: str = Form(...),  # JSON list string or comma-separated
    label_column: str = Form(...),
    weight_column: Optional[str] = Form(None)
):
    """
    Uploads a custom CSV dataset for model training in the sandboxed Lab zone.
    Validates file size (max 200MB), row cap (max 500k), schema mapping, and binary label.
    Stores dataset under data/lab/{dataset_id}/ with a server-generated UUID.
    """
    file_bytes = await file.read()

    # Parse feature_columns
    try:
        if feature_columns.strip().startswith("["):
            parsed_features = json.loads(feature_columns)
        else:
            parsed_features = [c.strip() for c in feature_columns.split(",") if c.strip()]
    except Exception:
        parsed_features = [c.strip() for c in feature_columns.split(",") if c.strip()]

    manifest = lab_dataset_service.validate_and_save_upload(
        file_bytes=file_bytes,
        original_filename=file.filename or "custom_dataset.csv",
        feature_columns=parsed_features,
        label_column=label_column.strip(),
        weight_column=weight_column.strip() if weight_column else None
    )
    return manifest


@router.get(
    "",
    response_model=LabDatasetListResponse,
    summary="List uploaded custom lab datasets"
)
def list_lab_datasets():
    """Lists all uploaded custom datasets and their manifests in the Lab zone."""
    datasets = lab_dataset_service.list_datasets()
    return LabDatasetListResponse(datasets=datasets)
