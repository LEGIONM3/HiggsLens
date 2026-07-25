"""
Dataset Upload Validation & Storage Service for HiggsLens Lab Sandboxed Zone.
"""

import hashlib
import io
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.lab import LabDatasetManifestSchema
from fastapi import HTTPException

logger = logging.getLogger("higgslens.lab.dataset_service")


class LabDatasetService:
    """
    Manages custom dataset uploads, schema validation, path-traversal safety,
    and storage under data/lab/{dataset_id}/.
    """

    def __init__(self, lab_data_dir: Optional[Path] = None):
        self.lab_data_dir = lab_data_dir or settings.LAB_DATA_DIR
        self.lab_data_dir.mkdir(parents=True, exist_ok=True)

    def validate_and_save_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        feature_columns: List[str],
        label_column: str,
        weight_column: Optional[str] = None
    ) -> LabDatasetManifestSchema:
        """
        Validates uploaded CSV file against size cap (200MB), row cap (500k),
        schema mapping, and binary label constraint.
        Generates server-side UUID dataset_id to prevent path traversal attacks.
        """
        # 1. Enforce file size cap (200 MB)
        if len(file_bytes) > settings.LAB_MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"Uploaded file exceeds maximum allowed size of {settings.LAB_MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
            )

        if len(file_bytes) == 0:
            raise HTTPException(status_code=422, detail="Uploaded CSV file is empty.")

        # 2. Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse CSV file: {str(e)}")

        # 3. Check row count cap (500,000)
        row_count = len(df)
        if row_count == 0:
            raise HTTPException(status_code=422, detail="Uploaded CSV dataset contains 0 rows.")
        if row_count > settings.LAB_MAX_DATASET_ROWS:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset row count ({row_count:,}) exceeds maximum allowed cap of {settings.LAB_MAX_DATASET_ROWS:,} rows."
            )

        # 4. Check column mapping
        all_cols = set(df.columns)
        missing_features = [c for c in feature_columns if c not in all_cols]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Specified feature columns not found in CSV: {missing_features}"
            )

        if label_column not in all_cols:
            raise HTTPException(
                status_code=422,
                detail=f"Specified label column '{label_column}' not found in CSV."
            )

        if weight_column and weight_column not in all_cols:
            raise HTTPException(
                status_code=422,
                detail=f"Specified weight column '{weight_column}' not found in CSV."
            )

        # 5. Check binary label constraint
        unique_labels = df[label_column].dropna().unique()
        if len(unique_labels) != 2:
            raise HTTPException(
                status_code=422,
                detail=f"Label column '{label_column}' must contain exactly 2 unique binary classes. Found {len(unique_labels)}: {list(unique_labels)[:5]}"
            )

        # 6. Generate UUID dataset_id and content hash
        dataset_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()

        # 7. Persist to storage data/lab/{dataset_id}/
        dataset_dir = self.lab_data_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        csv_path = dataset_dir / "dataset.csv"
        csv_path.write_bytes(file_bytes)

        manifest = LabDatasetManifestSchema(
            dataset_id=dataset_id,
            filename=Path(original_filename).name,  # basename sanitization
            row_count=row_count,
            column_count=len(df.columns),
            feature_columns=feature_columns,
            label_column=label_column,
            weight_column=weight_column if (weight_column and weight_column in all_cols) else None,
            created_at=created_at,
            content_hash=content_hash,
        )

        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        logger.info(f"Successfully uploaded custom lab dataset {dataset_id} ({row_count} rows)")
        return manifest

    def get_dataset_manifest(self, dataset_id: str) -> LabDatasetManifestSchema:
        """Retrieves manifest for requested dataset_id."""
        # Sanitize dataset_id string
        clean_id = Path(dataset_id).name
        manifest_path = self.lab_data_dir / clean_id / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"Lab dataset '{dataset_id}' not found.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LabDatasetManifestSchema(**data)

    def list_datasets(self) -> List[LabDatasetManifestSchema]:
        """Lists all uploaded custom lab dataset manifests."""
        datasets: List[LabDatasetManifestSchema] = []
        if not self.lab_data_dir.exists():
            return datasets

        for item in self.lab_data_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        datasets.append(LabDatasetManifestSchema(**data))
                    except Exception as e:
                        logger.warning(f"Error loading lab dataset manifest {manifest_path}: {e}")

        # Sort by creation time descending
        datasets.sort(key=lambda d: d.created_at, reverse=True)
        return datasets


lab_dataset_service = LabDatasetService()
