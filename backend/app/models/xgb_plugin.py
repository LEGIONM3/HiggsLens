from typing import Any, Optional

import numpy as np
import pandas as pd
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo


class XGBoostCandidate(ModelCandidate):
    def __init__(self):
        super().__init__(
            model_id="xgboost",
            display_name="XGBoost Classifier",
            required=False,
            preprocessing_strategy="native",
            supports_missing=True
        )
        self._status: Optional[str] = None

    def check_dependency(self) -> str:
        if self._status is not None:
            return self._status
        try:
            import xgboost as xgb
            self._status = "available"
        except ImportError:
            self._status = "unavailable"
        except Exception:
            self._status = "incompatible"
        return self._status

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.model_id,
            display_name=self.display_name,
            status=self.check_dependency(),
            required=self.required,
            supports_missing=self.supports_missing,
            preprocessing_pipeline=self.preprocessing_strategy,
            hyperparameters_schema={
                "n_estimators": {"type": "int", "default": 150, "description": "Number of boosting rounds"},
                "max_depth": {"type": "int", "default": 6, "description": "Maximum tree depth"},
                "learning_rate": {"type": "float", "default": 0.1, "description": "Learning rate"},
                "n_jobs": {"type": "int", "default": 4, "description": "CPU threads"}
            }
        )

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters: Any) -> "XGBoostCandidate":
        if self.check_dependency() != "available":
            raise RuntimeError("XGBoost is not installed or available in this runtime environment.")

        import xgboost as xgb
        n_estimators = int(hyperparameters.get("n_estimators", 150))
        max_depth = int(hyperparameters.get("max_depth", 6))
        learning_rate = float(hyperparameters.get("learning_rate", 0.1))
        n_jobs = int(hyperparameters.get("n_jobs", 4))
        seed = int(hyperparameters.get("random_state", 42))

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist"
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
