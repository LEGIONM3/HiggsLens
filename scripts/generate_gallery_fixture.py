"""
Offline script to generate certified gallery_fixture.json.
Uses certified PredictionService and XGBoost model artifact to score test-split events fixture.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict, cast

from backend.app.schemas.events import EventDataResponse
from backend.app.schemas.predict import PredictRequest
from backend.app.services.event_sampling import EventSamplingService
from backend.app.services.model_registry import ModelRegistryService
from backend.app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_gallery_fixture")


class ScoredItem(TypedDict):
  event: EventDataResponse
  prob: float
  label: str
  diff: float


def main():
  fixture_csv = (
      Path(__file__).resolve().parent.parent
      / "backend"
      / "tests"
      / "fixtures"
      / "events_fixture.csv"
  )
  sampling = EventSamplingService(data_path=fixture_csv)
  registry = ModelRegistryService()
  pred_service = PredictionService(registry=registry)

  artifact = registry.get_artifact("xgboost")
  threshold = float(artifact.metrics["optimal_threshold"])
  logger.info(f"Loaded certified XGBoost artifact threshold: {threshold}")

  # Fetch events from sampling service fixture
  sample_resp = sampling.sample_events(n=40, seed=42, label="any")
  test_events = sample_resp.events

  scored: List[ScoredItem] = []
  for ev in test_events:
    pred = pred_service.predict(
        PredictRequest(
            model_id="xgboost", features=ev.features, threshold=threshold
        )
    )
    prob = float(pred.signal_probability)
    scored.append({
        "event": ev,
        "prob": prob,
        "label": "signal" if pred.predicted_label == 1 else "background",
        "diff": abs(prob - threshold),
    })

  # Top Signal (highest prob)
  signal_sorted = sorted(scored, key=lambda x: x["prob"], reverse=True)
  top_signal = signal_sorted[:10]

  # Top Background (lowest prob)
  bg_sorted = sorted(scored, key=lambda x: x["prob"])
  top_background = bg_sorted[:10]

  # 10 Interesting cases (nearest threshold)
  qualifying_interesting = [e for e in scored if e["diff"] < 0.02]
  if len(qualifying_interesting) >= 10:
    qualifying_interesting.sort(key=lambda x: x["diff"])
    top_interesting = qualifying_interesting[:10]
    global_selection = "threshold_window"
  else:
    nearest_sorted = sorted(scored, key=lambda x: x["diff"])
    top_interesting = nearest_sorted[:10]
    global_selection = "nearest_threshold_fallback"

  events_out = []

  for rank, item in enumerate(top_signal, start=1):
    ev = item["event"]
    events_out.append({
        "event_id": ev.event_id,
        "features": ev.features,
        "signal_probability": round(item["prob"], 4),
        "predicted_label": item["label"],
        "threshold": round(threshold, 4),
        "gallery_category": "signal",
        "gallery_rank": rank,
        "selection_method": "top_probability",
    })

  for rank, item in enumerate(top_background, start=1):
    ev = item["event"]
    events_out.append({
        "event_id": ev.event_id,
        "features": ev.features,
        "signal_probability": round(item["prob"], 4),
        "predicted_label": item["label"],
        "threshold": round(threshold, 4),
        "gallery_category": "background",
        "gallery_rank": rank,
        "selection_method": "top_probability",
    })

  for rank, item in enumerate(top_interesting, start=1):
    ev = item["event"]
    method = (
        "threshold_window"
        if item["diff"] < 0.02
        else "nearest_threshold_fallback"
    )
    events_out.append({
        "event_id": ev.event_id,
        "features": ev.features,
        "signal_probability": round(item["prob"], 4),
        "predicted_label": item["label"],
        "threshold": round(threshold, 4),
        "gallery_category": "interesting",
        "gallery_rank": rank,
        "selection_method": method,
    })

  output = {
      "provenance": (
          "Generated offline from certified PredictionService and XGBoost model"
          " artifact (models/artifacts/xgboost/model.joblib) using"
          " backend/tests/fixtures/events_fixture.csv"
      ),
      "generation_command": (
          "uv run --python 3.12 python scripts/generate_gallery_fixture.py"
      ),
      "model_id": "xgboost",
      "threshold": round(threshold, 4),
      "events": events_out,
      "total_count": len(events_out),
      "categories": {
          "signal": len(top_signal),
          "background": len(top_background),
          "interesting": len(top_interesting),
      },
      "selection_method": global_selection,
  }

  out_p = (
      Path(__file__).resolve().parent.parent
      / "backend"
      / "tests"
      / "fixtures"
      / "gallery_fixture.json"
  )
  out_p.write_text(json.dumps(output, indent=2) + "\n")
  logger.info(
      f"Successfully generated certified gallery_fixture.json with"
      f" {len(events_out)} events at {out_p}"
  )


if __name__ == "__main__":
  main()
