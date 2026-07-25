import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.predict import PredictRequest, PredictResponse
from backend.app.services.model_registry import ModelRegistryService, model_registry_service

logger = logging.getLogger("higgslens.prediction_service")


class FeatureValidationError(Exception):
    """Raised when incoming feature vector fails schema validation."""
    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(message)


class PredictionService:
    """Service performing validated inference using pre-trained model artifacts."""

    def __init__(self, registry: Optional[ModelRegistryService] = None):
        self.registry = registry or model_registry_service

    def validate_features(self, features: Dict[str, float], schema: Dict[str, Any]) -> pd.DataFrame:
        feature_names = schema.get("feature_names", [])
        if not feature_names:
            raise FeatureValidationError("Invalid feature schema: missing feature_names list.")

        missing = [f for f in feature_names if f not in features]
        if missing:
            raise FeatureValidationError(
                f"Missing required feature(s): {missing}",
                details={"missing_features": missing}
            )

        row_data = []
        for name in feature_names:
            val = features[name]
            if not isinstance(val, (int, float, np.number)) or np.isnan(val):
                raise FeatureValidationError(
                    f"Invalid or non-numeric value for feature '{name}': {val}",
                    details={"feature": name, "value": val}
                )
            row_data.append(float(val))

        df = pd.DataFrame([row_data], columns=feature_names)
        return df

    def predict(self, request: PredictRequest) -> PredictResponse:
        artifact = self.registry.get_artifact(request.model_id)
        df = self.validate_features(request.features, artifact.feature_schema)
        model = self.registry.get_cached_model(request.model_id)

        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)
                if isinstance(probs, np.ndarray) and probs.ndim == 2:
                    prob_signal = float(probs[0, 1])
                else:
                    prob_signal = float(probs[0])
            elif hasattr(model, "predict"):
                preds = model.predict(df)
                prob_signal = float(preds[0])
            else:
                raise AttributeError("Loaded model object has neither predict_proba nor predict method.")
        except Exception as e:
            logger.error(f"Inference execution error for '{request.model_id}': {e}")
            raise RuntimeError(f"Error executing inference: {str(e)}")

        threshold = request.threshold
        if threshold is None:
            threshold = float(artifact.metrics.get("optimal_threshold", settings.DEFAULT_THRESHOLD))

        predicted_label = 1 if prob_signal >= threshold else 0

        return PredictResponse(
            signal_probability=prob_signal,
            predicted_label=predicted_label,
            threshold_used=threshold,
            model_id=request.model_id,
            manifest=artifact.manifest
        )


prediction_service = PredictionService()
