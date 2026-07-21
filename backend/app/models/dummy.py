from typing import Any

import numpy as np
import pandas as pd
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo
from sklearn.dummy import DummyClassifier


class DummyPriorCandidate(ModelCandidate):
    def __init__(self):
        super().__init__(
            model_id="dummy_prior",
            display_name="Dummy Classifier (Class Prior)",
            required=True,
            preprocessing_strategy="impute_median",
            supports_missing=False
        )

    def check_dependency(self) -> str:
        return "available"

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.model_id,
            display_name=self.display_name,
            status=self.check_dependency(),
            required=self.required,
            supports_missing=self.supports_missing,
            preprocessing_pipeline=self.preprocessing_strategy,
            hyperparameters_schema={
                "strategy": {"type": "string", "default": "prior", "description": "Strategy for predictions"}
            }
        )

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters: Any) -> "DummyPriorCandidate":
        strategy = hyperparameters.get("strategy", "prior")
        self.model = DummyClassifier(strategy=strategy)
        X_train = self.pipeline.fit_transform(df_train, feature_set)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X = self.pipeline.transform(df)
        return self.model.predict_proba(X)
