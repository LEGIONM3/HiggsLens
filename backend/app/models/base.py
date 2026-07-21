from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from backend.app.features.preprocessing import PreprocessingPipeline
from backend.app.schemas.models import ModelInfo


class ModelCandidate(ABC):
    """Abstract base class for all HiggsLens model candidates."""
    def __init__(self, model_id: str, display_name: str, required: bool, preprocessing_strategy: str, supports_missing: bool):
        self.model_id = model_id
        self.display_name = display_name
        self.required = required
        self.preprocessing_strategy = preprocessing_strategy
        self.supports_missing = supports_missing
        self.pipeline = PreprocessingPipeline(strategy=preprocessing_strategy)
        self.model: Any = None
        self.is_fitted: bool = False

    @abstractmethod
    def check_dependency(self) -> str:
        """Returns 'available', 'unavailable', or 'incompatible'."""
        pass

    @abstractmethod
    def get_info(self) -> ModelInfo:
        pass

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters: Any) -> "ModelCandidate":
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        pass

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(df)[:, 1]
        return (probs >= threshold).astype(np.int32)
