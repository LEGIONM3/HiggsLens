"""
Offline script to generate certified report_fixture.json.
Uses certified PredictionService, ExplanationService, and XGBoost model artifact
to score and explain test-split event 300000.
"""

import json
import logging
from pathlib import Path

from backend.app.schemas.explain import ExplainRequest
from backend.app.schemas.predict import PredictRequest
from backend.app.services.event_sampling import EventSamplingService
from backend.app.services.explanation import ExplanationService
from backend.app.services.model_registry import ModelRegistryService
from backend.app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_report_fixture")


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
  expl_service = ExplanationService(
      registry_service=registry,
      pred_service=pred_service,
      sampling_service=sampling,
  )

  artifact = registry.get_artifact("xgboost")
  threshold = float(artifact.metrics["optimal_threshold"])

  # Fetch test split event 300000
  ev = sampling.get_event_by_id(300000)
  if ev is None:
    raise RuntimeError("Event 300000 not found in fixture CSV!")

  # Predict
  pred_resp = pred_service.predict(
      PredictRequest(
          model_id="xgboost", features=ev.features, threshold=threshold
      )
  )

  # Explain
  expl_resp = expl_service.explain_event_by_id(300000, model_id="xgboost")

  # Read dataset manifest content hash
  manifest_p = (
      Path(__file__).resolve().parent.parent
      / "data"
      / "processed"
      / "v1"
      / "dataset_manifest.json"
  )
  v1_hash = "54242acf28a78ce303ea48bcf7002f0a44df08448271477e0a63331486c4f316"
  if manifest_p.exists():
    v1_hash = json.loads(manifest_p.read_text()).get("content_hash", v1_hash)

  report_fixture = {
      "provenance_header": (
          "Offline certified report fixture generated from Event 300000 using"
          " certified PredictionService, ExplanationService, and XGBoost"
          " model artifact."
      ),
      "generation_command": (
          "uv run --python 3.12 python scripts/generate_report_fixture.py"
      ),
      "event_id": 300000,
      "source_split": "test",
      "features": ev.features,
      "classification": {
          "model_id": "xgboost",
          "signal_probability": pred_resp.signal_probability,
          "predicted_label": (
              "signal" if pred_resp.predicted_label == 1 else "background"
          ),
          "threshold": threshold,
      },
      "explanation": {
          "base_value": expl_resp.base_value,
          "margin": expl_resp.margin,
          "attributions": [attr.model_dump() for attr in expl_resp.attributions],
          "object_groups": [
              grp.model_dump() for grp in expl_resp.object_groups
          ],
      },
      "gallery": {
          "category": "background",
          "rank": 1,
          "selection_method": "top_probability",
      },
      "reproducibility": {
          "model_artifact": {
              "model_id": "xgboost",
              "feature_schema_version": artifact.manifest.get(
                  "feature_schema_version", "v1"
              ),
              "device": artifact.manifest.get("device", "CPU (scikit-learn)"),
              "training_run_origin": artifact.manifest.get(
                  "training_run_origin", "Official baseline"
              ),
              "subsample_notes": artifact.manifest.get(
                  "subsample_notes", "Full dataset"
              ),
          },
          "dataset": {
              "record": "328",
              "doi": "10.7483/OPENDATA.ATLAS.ZBP2.M5T8",
              "content_hash": v1_hash,
          },
          "inference_contract": {
              "feature_count": 30,
              "sentinel_value": -999.0,
              "prediction_path": "certified PredictionService",
              "explanation_path": "native XGBoost pred_contribs=True",
          },
      },
      "provenance": {
          "statement": (
              "ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8)"
              " — official ATLAS simulated events, classified by certified"
              " pre-trained models."
          )
      },
  }

  out_p = (
      Path(__file__).resolve().parent.parent
      / "backend"
      / "tests"
      / "fixtures"
      / "report_fixture.json"
  )
  out_p.write_text(json.dumps(report_fixture, indent=2) + "\n")
  logger.info(f"Successfully generated certified report_fixture.json at {out_p}")


if __name__ == "__main__":
  main()
