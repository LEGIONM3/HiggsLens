from typing import List, Optional

import numpy as np
import pandas as pd
from backend.app.schemas.dataset import DERIVED_FEATURES, PRIMARY_FEATURES
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class PreprocessingPipeline:
    """
    Handles feature extraction, -999.0 sentinel processing, missingness indicators, and scaling.
    Ensures zero leakage across partitions by fitting imputers and scalers strictly on training data.
    """
    def __init__(self, strategy: str = "impute_median_scale", sentinel_value: float = -999.0):
        self.strategy = strategy  # "impute_median_scale", "impute_median", or "native"
        self.sentinel_value = sentinel_value
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def select_features(self, df: pd.DataFrame, feature_set: str) -> List[str]:
        if feature_set == "primary_only":
            cols = [c for c in PRIMARY_FEATURES if c in df.columns]
        elif feature_set == "derived_only":
            cols = [c for c in DERIVED_FEATURES if c in df.columns]
        else:  # all_physics
            cols = [c for c in (PRIMARY_FEATURES + DERIVED_FEATURES) if c in df.columns]
        return cols

    def fit(self, df: pd.DataFrame, feature_set: str = "all_physics") -> "PreprocessingPipeline":
        self.feature_names = self.select_features(df, feature_set)
        X = df[self.feature_names].copy()

        if self.strategy == "native":
            # For tree models with native missing value support, replace -999.0 with NaN
            self.is_fitted = True
            return self

        # Replace -999.0 with np.nan for imputer fitting
        X = X.replace(self.sentinel_value, np.nan)

        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.imputer.fit(X)

        if self.strategy == "impute_median_scale":
            X_imputed = self.imputer.transform(X)
            self.scaler = StandardScaler()
            self.scaler.fit(X_imputed)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted on training data before transforming.")

        X = df[self.feature_names].copy()
        if self.strategy == "native":
            return X.replace(self.sentinel_value, np.nan).to_numpy(dtype=np.float32)

        X = X.replace(self.sentinel_value, np.nan)
        if self.imputer is None:
            raise RuntimeError("Imputer not fitted on training data.")
        X_imputed = self.imputer.transform(X)

        if self.strategy == "impute_median_scale" and self.scaler is not None:
            return self.scaler.transform(X_imputed)
        return X_imputed

    def fit_transform(self, df: pd.DataFrame, feature_set: str = "all_physics") -> np.ndarray:
        return self.fit(df, feature_set).transform(df)
