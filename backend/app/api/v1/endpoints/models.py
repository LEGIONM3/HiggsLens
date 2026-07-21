import json
from typing import Any, Dict, Optional

from backend.app.core.config import settings
from backend.app.models.registry import model_registry
from backend.app.schemas.models import ModelInfo, ModelRegistryResponse
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/registry", response_model=ModelRegistryResponse)
def get_model_registry():
    models_dict = model_registry.list_models()
    # Convert dicts back to ModelInfo
    info_map = {m_id: ModelInfo.model_validate(data) for m_id, data in models_dict.items()}
    return ModelRegistryResponse(models=info_map)


@router.get("")
def list_models():
    return model_registry.list_models()


@router.get("/{model_id}/metrics")
def get_model_metrics(model_id: str):
    # Check if model exists in registry
    try:
        model_registry.get_candidate(model_id).get_info()
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Model ID '{model_id}' not found in registry.")

    # Search in experiment log or run json files for latest metrics for this model
    log_file = settings.ARTIFACTS_DIR / "experiments_log.jsonl"
    latest_metrics: Optional[Dict[str, Any]] = None

    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "metrics_by_model" in data and model_id in data["metrics_by_model"]:
                        latest_metrics = data["metrics_by_model"][model_id]
                except Exception:
                    continue

    if latest_metrics is None:
        return {"status": "untrained", "message": f"Model '{model_id}' has not been trained yet."}
    return latest_metrics
