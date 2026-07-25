"""
Model Registry Service for HiggsLens FastAPI Backend Service Layer.
Scans ARTIFACTS_DIR, validates manifests & metadata, and lazy-loads + caches trained weights.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from backend.app.core.config import settings

logger = logging.getLogger("higgslens.model_registry")


class ModelNotFoundError(Exception):
    """Raised when a requested model_id does not exist in the registry."""
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Unknown model_id '{model_id}'.")


class ArtifactCorruptError(Exception):
    """Raised when model weights or artifact files are missing or corrupt."""
    def __init__(self, model_id: str, message: str):
        self.model_id = model_id
        self.message = message
        super().__init__(f"Artifact missing or corrupt for model '{model_id}': {message}")


@dataclass
class ModelArtifact:
    model_id: str
    artifact_dir: Path
    manifest: Dict[str, Any]
    metrics: Dict[str, Any]
    feature_schema: Dict[str, Any]
    weights_path: Path
    cached_model: Optional[Any] = None

    @property
    def has_weights(self) -> bool:
        return self.weights_path.exists() and self.weights_path.is_file()


class ModelRegistryService:
    """
    Singleton Model Registry scanning ARTIFACTS_DIR, validating manifests & metadata,
    and lazy-loading + caching trained weights in memory for inference.
    Guards optional PyTorch dependencies cleanly.
    """
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or settings.ARTIFACTS_DIR
        self._artifacts: Dict[str, ModelArtifact] = {}
        self.scan_artifacts()

    def scan_artifacts(self) -> None:
        """Scans the artifacts directory and registers valid artifact contracts."""
        self._artifacts.clear()
        if not self.artifacts_dir.exists() or not self.artifacts_dir.is_dir():
            logger.warning(f"Artifacts directory {self.artifacts_dir} does not exist.")
            return

        for model_dir in self.artifacts_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_id = model_dir.name
            metrics_file = model_dir / "metrics.json"
            schema_file = model_dir / "feature_schema.json"
            manifest_file = model_dir / "manifest.json"

            # Check weights path: model.joblib or model.pt
            weights_file = model_dir / "model.joblib"
            if not weights_file.exists():
                weights_file = model_dir / "model.pt"

            if not (metrics_file.exists() and schema_file.exists() and manifest_file.exists()):
                logger.warning(f"Skipping incomplete artifact directory: {model_dir}")
                continue

            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                with open(schema_file, "r", encoding="utf-8") as f:
                    feature_schema = json.load(f)
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                artifact = ModelArtifact(
                    model_id=model_id,
                    artifact_dir=model_dir,
                    manifest=manifest,
                    metrics=metrics,
                    feature_schema=feature_schema,
                    weights_path=weights_file,
                    cached_model=None
                )
                self._artifacts[model_id] = artifact
                logger.info(f"Registered model artifact: {model_id} (weights_available={artifact.has_weights})")
            except Exception as e:
                logger.error(f"Error reading artifact for {model_id}: {e}")

    def list_model_ids(self) -> List[str]:
        return list(self._artifacts.keys())

    def get_artifact(self, model_id: str) -> ModelArtifact:
        if model_id not in self._artifacts:
            self.scan_artifacts()
        if model_id not in self._artifacts:
            raise ModelNotFoundError(model_id)
        return self._artifacts[model_id]

    def get_cached_model(self, model_id: str) -> Any:
        artifact = self.get_artifact(model_id)

        if not artifact.has_weights:
            raise ArtifactCorruptError(
                model_id,
                f"Model weights artifact is missing at {artifact.weights_path}"
            )

        if artifact.cached_model is None:
            try:
                logger.info(f"Lazy-loading model weights for '{model_id}' from {artifact.weights_path}")
                if artifact.weights_path.suffix == ".pt":
                    try:
                        import torch  # noqa: F401
                        artifact.cached_model = joblib.load(artifact.weights_path)
                    except ImportError:
                        raise ArtifactCorruptError(
                            model_id,
                            "PyTorch is not installed in the backend environment. Cannot load PyTorch model weights."
                        )
                else:
                    artifact.cached_model = joblib.load(artifact.weights_path)
            except ArtifactCorruptError:
                raise
            except Exception as e:
                raise ArtifactCorruptError(model_id, f"Failed to load weights: {str(e)}")

        return artifact.cached_model


model_registry_service = ModelRegistryService()
