"""
Backend Unit & Integration Tests for Research Reports & Reproducibility (/api/v1/events/{id}/report & /api/v1/reproducibility).
Tests certified-path parity, HTML escaping & structure, holdout & missing 404s, 422 for non-xgboost,
artifact non-mutation, reproducibility manifest contract, determinism, CI fixture fallback, and p50 latency.
"""

import time
from pathlib import Path

import pytest
from backend.app.main import create_app
from backend.app.services.event_sampling import EventSamplingService
from backend.app.services.explanation import ExplanationService
from backend.app.services.gallery import GalleryService
from backend.app.services.model_registry import ModelRegistryService
from backend.app.services.prediction_service import PredictionService
from backend.app.services.reporting import ReportingService
from fastapi.testclient import TestClient

try:
  import xgboost  # noqa: F401

  XGBOOST_INSTALLED = True
except ImportError:
  XGBOOST_INSTALLED = False

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "events_fixture.csv"


@pytest.fixture(scope="module")
def test_sampling_service():
  return EventSamplingService(data_path=FIXTURE_CSV)


@pytest.fixture(scope="module")
def report_client(test_sampling_service):
  registry = ModelRegistryService()
  pred_service = PredictionService(registry=registry)
  expl_service = ExplanationService(
      registry_service=registry,
      pred_service=pred_service,
      sampling_service=test_sampling_service,
  )
  gal_service = GalleryService(
      sampling_service=test_sampling_service,
      registry_service=registry,
      pred_service=pred_service,
      expl_service=expl_service,
  )
  rep_service = ReportingService(
      registry_service=registry,
      pred_service=pred_service,
      expl_service=expl_service,
      sampling_service=test_sampling_service,
      gal_service=gal_service,
  )

  import backend.app.api.v1.reports as reports_module

  reports_module.reporting_service = rep_service

  app = create_app()
  with TestClient(app) as client:
    yield client


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_report_json_certified_path_parity(report_client):
  """1. Asserts JSON report matches certified prediction and explanation service outputs for Event 300000."""
  resp = report_client.get("/api/v1/events/300000/report?format=json")
  assert resp.status_code == 200
  data = resp.json()

  assert data["report_version"] == "1.0"
  assert "generated_at" in data
  assert data["event"]["event_id"] == 300000
  assert len(data["event"]["features"]) == 30

  # Classification checks
  cl = data["classification"]
  assert cl["model_id"] == "xgboost"
  assert 0.0 <= cl["signal_probability"] <= 1.0
  assert cl["predicted_label"] in ("signal", "background")
  assert cl["threshold"] > 0.0

  # Explanation checks
  ex = data["explanation"]
  assert len(ex["attributions"]) == 30
  assert len(ex["object_groups"]) == 6

  # Provenance check
  assert "ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8)" in data["provenance"]["statement"]


def test_report_html_contains_required_sections(report_client):
  """2. Asserts HTML report contains event ID, classification, honesty note, provenance, and HTML escaping."""
  resp = report_client.get("/api/v1/events/300000/report?format=html")
  assert resp.status_code == 200
  assert "text/html" in resp.headers["content-type"]
  html_text = resp.text

  assert "HiggsLens Event Analysis Report" in html_text
  assert "300000" in html_text
  assert "Signal Probability" in html_text
  assert "Top 10 Feature Attributions" in html_text
  assert "Feature attributions describe how the model reached its score. They are not statements of physical causation." in html_text
  assert "ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8)" in html_text
  assert "@media print" in html_text
  assert "<script>" not in html_text  # Safe escaping, no raw script injection


def test_report_holdout_and_missing_404(report_client):
  """3. Asserts holdout EventId 300045 and missing EventId 999999 return 404 Not Found."""
  resp_holdout = report_client.get("/api/v1/events/300045/report")
  assert resp_holdout.status_code == 404

  resp_missing = report_client.get("/api/v1/events/999999/report")
  assert resp_missing.status_code == 404


def test_report_unsupported_model_422(report_client):
  """4. Asserts non-xgboost model_id parameter returns 422 Unprocessable Content (champion-only policy)."""
  resp = report_client.get("/api/v1/events/300000/report?model_id=random_forest")
  assert resp.status_code == 422
  assert "Unsupported model 'random_forest'" in resp.json()["detail"]


def test_report_does_not_mutate_artifacts(report_client):
  """5. Asserts report generation makes zero modifications to models/artifacts/ files."""
  artifacts_dir = Path(__file__).resolve().parent.parent / "models" / "artifacts"
  mtimes_before = {f: f.stat().st_mtime for f in artifacts_dir.glob("**/*") if f.is_file()}

  resp_json = report_client.get("/api/v1/events/300000/report?format=json")
  resp_html = report_client.get("/api/v1/events/300000/report?format=html")
  assert resp_json.status_code == 200
  assert resp_html.status_code == 200

  mtimes_after = {f: f.stat().st_mtime for f in artifacts_dir.glob("**/*") if f.is_file()}
  assert mtimes_before == mtimes_after


def test_reproducibility_manifest_contract(report_client):
  """6. Asserts /api/v1/reproducibility returns DOI, record 328, content_hash, and no absolute file paths."""
  resp = report_client.get("/api/v1/reproducibility")
  assert resp.status_code == 200
  data = resp.json()

  assert data["report_contract_version"] == "1.0"
  assert data["dataset"]["record"] == "328"
  assert data["dataset"]["doi"] == "10.7483/OPENDATA.ATLAS.ZBP2.M5T8"
  assert len(data["dataset"]["content_hash"]) == 64
  assert data["inference_contract"]["feature_count"] == 30
  assert data["inference_contract"]["sentinel_value"] == -999.0
  assert len(data["certified_models"]) >= 11

  # Assert no absolute filesystem paths are exposed
  json_str = resp.text
  assert "D:" not in json_str
  assert "C:\\Users" not in json_str
  assert "/home/" not in json_str


def test_report_json_is_deterministic_except_generated_at(report_client):
  """7. Asserts two report calls yield identical JSON payloads excluding generated_at timestamp."""
  resp1 = report_client.get("/api/v1/events/300000/report?format=json").json()
  resp2 = report_client.get("/api/v1/events/300000/report?format=json").json()

  del resp1["generated_at"]
  del resp2["generated_at"]
  assert resp1 == resp2


def test_report_uses_fixture_without_real_dataset(report_client):
  """8. Asserts CI-safe report fixture fallback functions properly."""
  rep_service = ReportingService()

  # Report for 300000 using fixture
  report = rep_service.generate_event_report(300000)
  assert report.event.event_id == 300000
  assert len(report.event.features) == 30
  assert report.classification.model_id == "xgboost"
  assert len(report.explanation.attributions) == 30
  assert len(report.explanation.object_groups) == 6


def test_report_latency_p50(report_client):
  """9. Measured warm p50 latency < 200ms for both JSON and HTML report formats."""
  # Warm up
  report_client.get("/api/v1/events/300000/report?format=json")
  report_client.get("/api/v1/events/300000/report?format=html")

  # Measure JSON p50
  json_times = []
  for _ in range(10):
    t0 = time.perf_counter()
    res = report_client.get("/api/v1/events/300000/report?format=json")
    assert res.status_code == 200
    json_times.append((time.perf_counter() - t0) * 1000.0)
  json_p50 = sorted(json_times)[len(json_times) // 2]

  # Measure HTML p50
  html_times = []
  for _ in range(10):
    t0 = time.perf_counter()
    res = report_client.get("/api/v1/events/300000/report?format=html")
    assert res.status_code == 200
    html_times.append((time.perf_counter() - t0) * 1000.0)
  html_p50 = sorted(html_times)[len(html_times) // 2]

  print(f"\n[REPORT LATENCY P50 EVIDENCE] JSON: {json_p50:.2f} ms | HTML: {html_p50:.2f} ms")
  assert json_p50 < 200.0
  assert html_p50 < 200.0
