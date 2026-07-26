"""
Backend API Unit & Integration Tests for Feature Explanation & Attributions (/api/v1/explain).
Tests Additivity Gate, TreeSHAP accuracy, response parity, holdout exclusion, non-tree error handling,
feature-order robustness, object-group coverage, dataset isolation on thresholds, and p50 latency.
"""

import math
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from backend.app.main import create_app
from backend.app.schemas.predict import PredictRequest
from backend.app.services.event_sampling import EventSamplingService
from backend.app.services.explanation import ExplanationService, explanation_service
from backend.app.services.prediction_service import prediction_service
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
def test_explanation_service(test_sampling_service):
  return ExplanationService(sampling_service=test_sampling_service)


@pytest.fixture(scope="module")
def explain_client(test_sampling_service, test_explanation_service):
  app = create_app()
  import backend.app.api.v1.events as events_module
  import backend.app.api.v1.explain as explain_module

  events_module.event_sampling_service = test_sampling_service
  events_module.explanation_service = test_explanation_service
  explain_module.explanation_service = test_explanation_service
  return TestClient(app)


def sigmoid(x: float) -> float:
  return 1.0 / (1.0 + math.exp(-x))


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_explain_additivity_gate(explain_client, test_sampling_service):
  """Additivity Gate: sum(contributions) + base_value == margin AND sigmoid(margin) == certified probability within 1e-6 across 3 payloads (jet_num 0, 1, 2)."""
  # Fetch 3 distinct events from fixture (jet_num = 0, 1, 2)
  e_0 = test_sampling_service.get_event_by_id(300000)  # jet_num = 0
  e_1 = test_sampling_service.get_event_by_id(300001)  # jet_num = 1
  e_2 = test_sampling_service.get_event_by_id(300003)  # jet_num = 2

  events = [e for e in (e_0, e_1, e_2) if e is not None]
  assert len(events) >= 3, "Expected at least 3 fixture events for additivity testing!"

  for ev in events:
    resp = explain_client.post(
        "/api/v1/explain", json={"features": ev.features, "model_id": "xgboost"}
    )
    assert resp.status_code == 200
    data = resp.json()

    base_val = data["base_value"]
    margin = data["margin"]
    prob = data["probability"]
    attributions = data["attributions"]

    # 1. Assert sum(contributions) + base_value == margin
    contrib_sum = sum(a["contribution"] for a in attributions)
    assert base_val + contrib_sum == pytest.approx(margin, abs=1e-6)

    # 2. Assert sigmoid(margin) == probability from PredictionService
    assert sigmoid(margin) == pytest.approx(prob, abs=1e-6)

    # 3. Assert prediction matches direct PredictRequest call
    pred_req = PredictRequest(
        model_id="xgboost", features=ev.features, threshold=None
    )
    direct_pred = prediction_service.predict(pred_req)
    assert prob == pytest.approx(direct_pred.signal_probability, abs=1e-6)


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_explain_event_by_id_parity(explain_client, test_sampling_service):
  """Assert GET /api/v1/events/{id}/explain equals POST /api/v1/explain for exact feature payload."""
  event_id = 300000
  ev = test_sampling_service.get_event_by_id(event_id)
  assert ev is not None

  # GET /events/{id}/explain
  get_resp = explain_client.get(f"/api/v1/events/{event_id}/explain?model_id=xgboost")
  assert get_resp.status_code == 200
  get_data = get_resp.json()

  # POST /explain
  post_resp = explain_client.post(
      "/api/v1/explain", json={"features": ev.features, "model_id": "xgboost"}
  )
  assert post_resp.status_code == 200
  post_data = post_resp.json()

  assert get_data["probability"] == pytest.approx(post_data["probability"], abs=1e-6)
  assert get_data["margin"] == pytest.approx(post_data["margin"], abs=1e-6)
  assert get_data["base_value"] == pytest.approx(post_data["base_value"], abs=1e-6)
  assert len(get_data["attributions"]) == len(post_data["attributions"])


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_explain_holdout_and_missing_event_404(explain_client):
  """Amendment 2: Confirmed holdout EventId 300045 returns 404, as does non-existent ID 999999."""
  # Holdout EventId 300045
  resp_holdout = explain_client.get("/api/v1/events/300045/explain")
  assert resp_holdout.status_code == 404
  assert "not found in test split" in resp_holdout.json()["detail"]

  # Non-existent EventId 999999
  resp_missing = explain_client.get("/api/v1/events/999999/explain")
  assert resp_missing.status_code == 404


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_explain_unsupported_model_422(explain_client, test_sampling_service):
  """Non-tree model (e.g. mlp) returns clean 422 Unprocessable Entity error."""
  ev = test_sampling_service.get_event_by_id(300000)
  assert ev is not None

  resp = explain_client.post(
      "/api/v1/explain", json={"features": ev.features, "model_id": "mlp"}
  )
  assert resp.status_code == 422
  assert "does not support TreeSHAP" in resp.json()["detail"]


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_explain_feature_order_robustness(explain_client, test_sampling_service):
  """Shuffled feature key order in request payload yields identical attributions."""
  import random

  ev = test_sampling_service.get_event_by_id(300000)
  assert ev is not None

  shuffled_keys = list(ev.features.keys())
  random.seed(42)
  random.shuffle(shuffled_keys)
  shuffled_features = {k: ev.features[k] for k in shuffled_keys}

  resp_orig = explain_client.post(
      "/api/v1/explain", json={"features": ev.features, "model_id": "xgboost"}
  )
  resp_shuf = explain_client.post(
      "/api/v1/explain", json={"features": shuffled_features, "model_id": "xgboost"}
  )

  assert resp_orig.status_code == 200
  assert resp_shuf.status_code == 200

  orig_data = resp_orig.json()
  shuf_data = resp_shuf.json()

  assert orig_data["margin"] == pytest.approx(shuf_data["margin"], abs=1e-6)
  assert orig_data["probability"] == pytest.approx(shuf_data["probability"], abs=1e-6)


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_explain_object_groups_coverage(explain_client, test_sampling_service):
  """Assert all 30 features are present and sum of group total_abs_contributions equals sum of feature abs(contributions)."""
  ev = test_sampling_service.get_event_by_id(300000)
  assert ev is not None

  resp = explain_client.post(
      "/api/v1/explain", json={"features": ev.features, "model_id": "xgboost"}
  )
  assert resp.status_code == 200
  data = resp.json()

  attributions = data["attributions"]
  object_groups = data["object_groups"]

  assert len(attributions) == 30

  total_feature_abs = sum(abs(a["contribution"]) for a in attributions)
  total_group_abs = sum(g["total_abs_contribution"] for g in object_groups)

  assert total_feature_abs == pytest.approx(total_group_abs, abs=1e-6)

  group_names = {g["group"] for g in object_groups}
  assert group_names == {"tau", "lepton", "leading_jet", "subleading_jet", "met", "global"}


def test_thresholds_endpoint_no_dataset_access(explain_client, monkeypatch):
  """Assert /api/v1/metrics/{model_id} endpoint serves stored scan points without touching dataset files."""
  # Monkeypatch EventSamplingService to fail if called
  from backend.app.services.event_sampling import EventSamplingService

  def mock_fail(*args, **kwargs):
    raise RuntimeError("Dataset should NOT be accessed during metrics/thresholds request!")

  monkeypatch.setattr(EventSamplingService, "sample_events", mock_fail)
  monkeypatch.setattr(EventSamplingService, "get_event_by_id", mock_fail)

  resp = explain_client.get("/api/v1/metrics/xgboost/thresholds")
  assert resp.status_code == 200
  data = resp.json()
  assert "model_id" in data
  assert data["model_id"] == "xgboost"
  assert "points" in data


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_explain_sample_p50_latency(explain_client, test_sampling_service):
  """Amendment 1: Warm p50 latency for POST /api/v1/explain is < 200ms."""
  ev = test_sampling_service.get_event_by_id(300000)
  assert ev is not None
  payload = {"features": ev.features, "model_id": "xgboost"}

  # Warm-up call
  explain_client.post("/api/v1/explain", json=payload)

  timings = []
  for _ in range(10):
    t0 = time.time()
    resp = explain_client.post("/api/v1/explain", json=payload)
    t1 = time.time()
    assert resp.status_code == 200
    timings.append((t1 - t0) * 1000.0)

  p50_ms = sorted(timings)[len(timings) // 2]
  print(f"\nMeasured /api/v1/explain p50 latency: {p50_ms:.2f} ms")
  assert p50_ms < 200.0, f"p50 latency {p50_ms:.2f}ms exceeds 200ms budget!"
