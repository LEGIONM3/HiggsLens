from typing import Any

import numpy as np
import pandas as pd
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo
from sklearn.ensemble import HistGradientBoostingClassifier


class HistGradientBoostingCandidate(ModelCandidate):
    def __init__(self):
        super().__init__(
            model_id="histogram_gradient_boosting",
            display_name="Histogram Gradient Boosting Classifier",
            required=True,
            preprocessing_strategy="native",
            supports_missing=True
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
                "max_iter": {"type": "int", "default": 150, "description": "Maximum boosting iterations"},
                "max_depth": {"type": "int", "default": 10, "description": "Maximum tree depth"},
                "learning_rate": {"type": "float", "default": 0.1, "description": "Boosting learning rate"}
            }
        )

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters: Any) -> "HistGradientBoostingCandidate":
        max_iter = int(hyperparameters.get("max_iter", 150))
        max_depth = int(hyperparameters.get("max_depth", 10))
        learning_rate = float(hyperparameters.get("learning_rate", 0.1))
        seed = int(hyperparameters.get("random_state", 42))

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed
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
