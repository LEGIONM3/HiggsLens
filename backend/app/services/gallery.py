"""
Curated Event Gallery Service for HiggsLens (/api/v1/events/gallery).
Serves top 20 signal, top 20 background, and 10 near-threshold interesting cases from test split.
Supports CI fallback to committed gallery_fixture.json when real dataset or XGBoost is absent.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

import numpy as np

from app.core.config import settings
from app.schemas.events import EventDataResponse
from app.schemas.explain import ExplainRequest
from app.schemas.gallery import (
  GalleryEventSummary,
  GalleryResponse,
  PermalinkResponse,
)
from app.schemas.predict import PredictRequest
from app.services.event_sampling import (
  EventSamplingService,
  event_sampling_service,
)
from app.services.explanation import ExplanationService, explanation_service
from app.services.model_registry import (
  ModelRegistryService,
  model_registry_service,
)
from app.services.prediction_service import (
  PredictionService,
  prediction_service,
)


class ScoredGalleryItem(TypedDict):
  event: EventDataResponse
  prob: float
  label: str
  diff: float

logger = logging.getLogger("higgslens.gallery_service")

FIXTURE_GALLERY_JSON = (
    Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "fixtures"
    / "gallery_fixture.json"
)


class GalleryService:
  """Service for managing curated event gallery and permalinks."""

  def __init__(
      self,
      sampling_service: Optional[EventSamplingService] = None,
      pred_service: Optional[PredictionService] = None,
      expl_service: Optional[ExplanationService] = None,
      registry_service: Optional[ModelRegistryService] = None,
  ):
    self.sampling_service = sampling_service or event_sampling_service
    self.pred_service = pred_service or prediction_service
    self.expl_service = expl_service or explanation_service
    self.registry_service = registry_service or model_registry_service
    self._cached_gallery: Optional[GalleryResponse] = None

  def get_gallery(
      self, model_id: str = "xgboost", force_recompute: bool = False
  ) -> GalleryResponse:
    """Returns curated gallery response, using memory cache or building on demand."""
    if self._cached_gallery is not None and not force_recompute:
      return self._cached_gallery

    # Fallback to fixture if dataset or XGBoost model is unavailable (CI environment)
    try:
      gallery = self._compute_live_gallery(model_id=model_id)
      self._cached_gallery = gallery
      return gallery
    except Exception as e:
      logger.info(
        f"Live gallery calculation unavailable ({e}); loading CI fixture"
        f" fallback {FIXTURE_GALLERY_JSON.name}"
      )
      return self._load_fixture_gallery()

  def _compute_live_gallery(self, model_id: str = "xgboost") -> GalleryResponse:
    """Computes gallery from test split events using certified model predictions."""
    # 1. Fetch headline metrics for threshold
    artifact = self.registry_service.get_artifact(model_id)
    threshold = float(
        artifact.metrics.get("optimal_threshold", settings.DEFAULT_THRESHOLD)
    )

    # 2. Fetch test split events (sample up to 1000 events for gallery building)
    sample_resp = self.sampling_service.sample_events(
        n=50, seed=42, label="any"
    )
    if not sample_resp.events:
      raise RuntimeError("No test split events available for gallery.")

    # 3. Score all events via certified PredictionService
    scored_events: List[ScoredGalleryItem] = []
    for ev in sample_resp.events:
      pred = self.pred_service.predict(
          PredictRequest(
              model_id=model_id, features=ev.features, threshold=threshold
          )
      )
      scored_events.append({
          "event": ev,
          "prob": pred.signal_probability,
          "label": (
              "signal" if pred.predicted_label == 1 else "background"
          ),
          "diff": abs(pred.signal_probability - threshold),
      })

    # Sort categories
    # Top 20 Signal: highest signal_probability
    signal_sorted = sorted(
        scored_events, key=lambda x: x["prob"], reverse=True
    )
    top_signal = signal_sorted[:20]

    # Top 20 Background: lowest signal_probability
    background_sorted = sorted(scored_events, key=lambda x: x["prob"])
    top_background = background_sorted[:20]

    # 10 Interesting Cases: near threshold (|prob - threshold| < 0.02)
    qualifying_interesting = [
        e for e in scored_events if e["diff"] < 0.02
    ]
    if len(qualifying_interesting) >= 10:
      qualifying_interesting.sort(key=lambda x: x["diff"])
      top_interesting = qualifying_interesting[:10]
      global_selection = "threshold_window"
    else:
      # Amendment 3: Nearest threshold fallback if fewer than 10 match < 0.02
      nearest_sorted = sorted(scored_events, key=lambda x: x["diff"])
      top_interesting = nearest_sorted[:10]
      global_selection = "nearest_threshold_fallback"

    # Assemble GalleryEventSummary items
    gallery_items: List[GalleryEventSummary] = []

    for rank, item in enumerate(top_signal, start=1):
      ev = item["event"]
      gallery_items.append(
          GalleryEventSummary(
              event_id=ev.event_id,
              features=ev.features,
              signal_probability=item["prob"],
              predicted_label=item["label"],
              threshold=threshold,
              gallery_category="signal",
              gallery_rank=rank,
              selection_method="top_probability",
          )
      )

    for rank, item in enumerate(top_background, start=1):
      ev = item["event"]
      gallery_items.append(
          GalleryEventSummary(
              event_id=ev.event_id,
              features=ev.features,
              signal_probability=item["prob"],
              predicted_label=item["label"],
              threshold=threshold,
              gallery_category="background",
              gallery_rank=rank,
              selection_method="top_probability",
          )
      )

    for rank, item in enumerate(top_interesting, start=1):
      ev = item["event"]
      method = (
          "threshold_window"
          if item["diff"] < 0.02
          else "nearest_threshold_fallback"
      )
      gallery_items.append(
          GalleryEventSummary(
              event_id=ev.event_id,
              features=ev.features,
              signal_probability=item["prob"],
              predicted_label=item["label"],
              threshold=threshold,
              gallery_category="interesting",
              gallery_rank=rank,
              selection_method=method,
          )
      )

    categories_count = {
        "signal": len(top_signal),
        "background": len(top_background),
        "interesting": len(top_interesting),
    }

    return GalleryResponse(
        events=gallery_items,
        total_count=len(gallery_items),
        categories=categories_count,
        selection_method=global_selection,
    )

  def _load_fixture_gallery(self) -> GalleryResponse:
    """Loads committed fixture sidecar gallery_fixture.json for CI environments."""
    if not FIXTURE_GALLERY_JSON.exists():
      raise FileNotFoundError(f"Fixture gallery file missing at {FIXTURE_GALLERY_JSON}")

    data = json.loads(FIXTURE_GALLERY_JSON.read_text())
    events = [GalleryEventSummary(**e) for e in data["events"]]
    return GalleryResponse(
        events=events,
        total_count=data.get("total_count", len(events)),
        categories=data.get("categories", {"signal": 10, "background": 10, "interesting": 10}),
        selection_method=data.get("selection_method", "nearest_threshold_fallback"),
    )

  def get_gallery_event_by_id(
      self, event_id: int, model_id: str = "xgboost"
  ) -> Optional[GalleryEventSummary]:
    """Returns single gallery event by ID, or None if not present in gallery."""
    gallery = self.get_gallery(model_id=model_id)
    for ev in gallery.events:
      if ev.event_id == event_id:
        return ev
    return None

  def get_permalink(
      self, event_id: int, model_id: str = "xgboost"
  ) -> PermalinkResponse:
    """Returns combined permalink response (features + predict + TreeSHAP explanation) in a single API call."""
    event = self.sampling_service.get_event_by_id(event_id)
    if event is None:
      raise KeyError(f"EventId {event_id} not found in test split or is holdout.")

    # Single call to explanation service
    explanation = self.expl_service.explain_event_by_id(event_id, model_id=model_id)

    return PermalinkResponse(
        event_id=event.event_id,
        features=event.features,
        kaggleset="v",
        signal_probability=explanation.probability,
        predicted_label=explanation.predicted_label,
        threshold=explanation.threshold,
        explanation=explanation,
    )


gallery_service = GalleryService()
