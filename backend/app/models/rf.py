from typing import Any

import numpy as np
import pandas as pd
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo
from sklearn.ensemble import RandomForestClassifier


class RandomForestCandidate(ModelCandidate):
    def __init__(self):
        super().__init__(
            model_id="random_forest",
            display_name="Random Forest Classifier",
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
                "n_estimators": {"type": "int", "default": 100, "description": "Number of trees"},
                "max_depth": {"type": "int", "default": 15, "description": "Maximum tree depth"},
                "n_jobs": {"type": "int", "default": 4, "description": "Parallel CPU threads"}
            }
        )

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters: Any) -> "RandomForestCandidate":
        n_estimators = int(hyperparameters.get("n_estimators", 100))
        max_depth = int(hyperparameters.get("max_depth", 15))
        n_jobs = int(hyperparameters.get("n_jobs", 4))
        seed = int(hyperparameters.get("random_state", 42))

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=seed,
            max_samples=0.8
        )
        X_train = self.pipeline.fit_transform(df_train, feature_set)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X = self.pipeline.transform(df)
        return self.model.predict_proba(X)
