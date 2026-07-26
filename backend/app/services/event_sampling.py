"""
Event Sampling Service for HiggsLens (/api/v1/events).
Loads test-split events, enforces holdout exclusion, and routes inference
through the certified PredictionService code path.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.events import (
    EventDataResponse,
    EventPredictionResponse,
    EventSampleResponse,
)
from backend.app.schemas.predict import PredictRequest
from backend.app.services.prediction_service import (
    PredictionService,
    prediction_service,
)
from ml.data.feature_sets import ALL_PHYSICS_FEATURES

logger = logging.getLogger("higgslens.event_sampling")


class EventDatasetNotFoundError(Exception):
    """Raised when the requested test dataset CSV is not found at runtime."""
    def __init__(self, path: Path):
        self.path = path
        super().__init__(
            f"Test split dataset unavailable at path '{path}'. "
            "Dataset pipeline must be run first."
        )


class EventSamplingService:
    """
    Service for loading & sampling real ATLAS test-split collision events.
    Enforces strict holdout isolation and reuses the certified PredictionService path.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        pred_service: Optional[PredictionService] = None,
        model_id: str = "xgboost"
    ):
        self._custom_data_path = data_path
        self.pred_service = pred_service or prediction_service
        self.model_id = model_id
        self._df: Optional[pd.DataFrame] = None
        self._id_index: Dict[int, int] = {}

    @property
    def data_path(self) -> Path:
        if self._custom_data_path is not None:
            return self._custom_data_path
        return settings.DATA_DIR / "processed" / "v1" / "test.csv"

    def _load_data(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        target_path = self.data_path
        if not target_path.exists() or not target_path.is_file():
            logger.warning(f"Event sampling dataset missing at {target_path}")
            raise EventDatasetNotFoundError(target_path)

        logger.info(f"Loading event sampling test dataset from {target_path}...")
        # Read required columns only for memory efficiency
        cols_to_read = ["EventId", "Label", "KaggleSet"] + list(ALL_PHYSICS_FEATURES)
        df = pd.read_csv(target_path, usecols=lambda c: c in cols_to_read)

        # STRICT HOLDOUT EXCLUSION: Filter test split ('v') ONLY.
        if "KaggleSet" in df.columns:
            df = df[df["KaggleSet"] == "v"].reset_index(drop=True)

        assert len(df) > 0, "No test split events found after KaggleSet=='v' filtering!"

        self._df = df
        # Index EventId -> DataFrame index for O(1) lookup
        self._id_index = {int(row.EventId): i for i, row in df.iterrows()}
        logger.info(f"Loaded {len(df):,} test split events into EventSamplingService cache.")
        return self._df

    def sample_events(
        self,
        n: int = 12,
        seed: int = 42,
        label: str = "any"
    ) -> EventSampleResponse:

        if n <= 0 or n > 50:
            raise ValueError(f"Sample size n must be between 1 and 50 (got {n}).")

        df = self._load_data()
        label_clean = (label or "any").strip().lower()

        if label_clean in ("signal", "s"):
            filtered_df = df[df["Label"] == "s"]
        elif label_clean in ("background", "b"):
            filtered_df = df[df["Label"] == "b"]
        else:
            filtered_df = df

        if len(filtered_df) == 0:
            raise ValueError(f"No events match label filter '{label}'.")

        sample_n = min(n, len(filtered_df))
        rng = np.random.RandomState(seed)
        sampled_indices = rng.choice(filtered_df.index.to_numpy(), size=sample_n, replace=False)

        event_responses: List[EventDataResponse] = []
        for idx in sampled_indices:
            row = df.iloc[idx]
            event_id = int(row["EventId"])
            label_char = str(row["Label"])
            true_label = "signal" if label_char == "s" else "background"

            features_dict = {f: float(row[f]) for f in ALL_PHYSICS_FEATURES}

            # REUSE CERTIFIED PREDICT PATH (Amendment 2 & Amendment 3)
            predict_req = PredictRequest(
                model_id=self.model_id,
                features=features_dict,
                threshold=None
            )
            pred_resp = self.pred_service.predict(predict_req)

            pred_label_str = "signal" if pred_resp.predicted_label == 1 else "background"

            event_responses.append(
                EventDataResponse(
                    event_id=event_id,
                    true_label=true_label,
                    features=features_dict,
                    prediction=EventPredictionResponse(
                        model_id=self.model_id,
                        probability=pred_resp.signal_probability,
                        predicted_label=pred_label_str,
                        threshold=pred_resp.threshold_used
                    )
                )
            )

        return EventSampleResponse(
            events=event_responses,
            count=len(event_responses),
            seed=seed,
            label_filter=label_clean
        )

    def get_event_by_id(self, event_id: int) -> Optional[EventDataResponse]:
        df = self._load_data()
        if event_id not in self._id_index:
            return None

        idx = self._id_index[event_id]
        row = df.iloc[idx]

        label_char = str(row["Label"])
        true_label = "signal" if label_char == "s" else "background"
        features_dict = {f: float(row[f]) for f in ALL_PHYSICS_FEATURES}

        # REUSE CERTIFIED PREDICT PATH (Amendment 2 & Amendment 3)
        predict_req = PredictRequest(
            model_id=self.model_id,
            features=features_dict,
            threshold=None
        )
        pred_resp = self.pred_service.predict(predict_req)
        pred_label_str = "signal" if pred_resp.predicted_label == 1 else "background"

        return EventDataResponse(
            event_id=event_id,
            true_label=true_label,
            features=features_dict,
            prediction=EventPredictionResponse(
                model_id=self.model_id,
                probability=pred_resp.signal_probability,
                predicted_label=pred_label_str,
                threshold=pred_resp.threshold_used
            )
        )


event_sampling_service = EventSamplingService()
