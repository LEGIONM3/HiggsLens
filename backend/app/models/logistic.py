from typing import Any

import numpy as np
import pandas as pd
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo
from sklearn.linear_model import LogisticRegression


class LogisticRegressionCandidate(ModelCandidate):
    def __init__(self):
        super().__init__(
            model_id="logistic_regression",
            display_name="Logistic Regression",
            required=True,
            preprocessing_strategy="impute_median_scale",
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
                "C": {"type": "float", "default": 1.0, "description": "Inverse regularization strength"},
                "max_iter": {"type": "int", "default": 1000, "description": "Maximum optimization iterations"}
            }
        )

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters: Any) -> "LogisticRegressionCandidate":
        C = float(hyperparameters.get("C", 1.0))
        max_iter = int(hyperparameters.get("max_iter", 1000))
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=42)
        X_train = self.pipeline.fit_transform(df_train, feature_set)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X = self.pipeline.transform(df)
        return self.model.predict_proba(X)
