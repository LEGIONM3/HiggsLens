"""
Backend API Unit & Integration Tests for Curated Event Gallery & Permalinks (/api/v1/events/gallery & /permalink).
Tests category structure, holdout exclusion, static route ordering, fixture fallback, single permalink response shape,
interesting fallback metadata, and p50 latency performance.
"""

import time
from pathlib import Path

import pytest
from backend.app.main import create_app
from backend.app.services.event_sampling import EventSamplingService
from backend.app.services.explanation import ExplanationService
from backend.app.services.gallery import GalleryService
from fastapi.testclient import TestClient

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "events_fixture.csv"


@pytest.fixture(scope="module")
def test_sampling_service():
  return EventSamplingService(data_path=FIXTURE_CSV)


@pytest.fixture(scope="module")
def test_explanation_service(test_sampling_service):
  return ExplanationService(sampling_service=test_sampling_service)


@pytest.fixture(scope="module")
def test_gallery_service(test_sampling_service, test_explanation_service):
  return GalleryService(
      sampling_service=test_sampling_service,
      expl_service=test_explanation_service,
  )


@pytest.fixture(scope="module")
def gallery_client(test_sampling_service, test_explanation_service, test_gallery_service):
  app = create_app()
  import backend.app.api.v1.events as events_module

  events_module.event_sampling_service = test_sampling_service
  events_module.explanation_service = test_explanation_service
  events_module.gallery_service = test_gallery_service
  return TestClient(app)


def test_gallery_returns_three_categories(gallery_client):
  """Assert /api/v1/events/gallery returns 'signal', 'background', and 'interesting' categories."""
  resp = gallery_client.get("/api/v1/events/gallery")
  assert resp.status_code == 200
  data = resp.json()

  assert "events" in data
  assert "categories" in data
  categories = data["categories"]

  assert "signal" in categories
  assert "background" in categories
  assert "interesting" in categories
  assert data["total_count"] > 0


def test_gallery_category_counts(gallery_client):
  """Assert category counts obey maximum bounds (signal <= 20, background <= 20, interesting <= 10)."""
  resp = gallery_client.get("/api/v1/events/gallery")
  assert resp.status_code == 200
  data = resp.json()

  cats = data["categories"]
  assert cats["signal"] <= 20
  assert cats["background"] <= 20
  assert cats["interesting"] <= 10


def test_gallery_event_by_id(gallery_client):
  """Amendment 1 & 2: Static /api/v1/events/gallery/{id} returns single gallery event without route collision."""
  # First get gallery list
  res_list = gallery_client.get("/api/v1/events/gallery")
  events = res_list.json()["events"]
  assert len(events) > 0

  target_id = events[0]["event_id"]
  resp = gallery_client.get(f"/api/v1/events/gallery/{target_id}")
  assert resp.status_code == 200
  data = resp.json()

  assert data["event_id"] == target_id
  assert "gallery_category" in data
  assert "gallery_rank" in data


def test_gallery_holdout_excluded(gallery_client):
  """Assert zero gallery events belong to holdout split (KaggleSet == 'u')."""
  resp = gallery_client.get("/api/v1/events/gallery")
  assert resp.status_code == 200
  events = resp.json()["events"]

  # Holdout fixture EventIds are 300040 to 300049
  holdout_ids = set(range(300040, 300050))
  for ev in events:
    assert ev["event_id"] not in holdout_ids


def test_permalink_response_shape(gallery_client, test_sampling_service):
  """Assert GET /api/v1/events/{id}/permalink returns combined event + explanation payload in one call."""
  # Event 300000 in test split
  ev = test_sampling_service.get_event_by_id(300000)
  assert ev is not None

  resp = gallery_client.get("/api/v1/events/300000/permalink?model_id=xgboost")
  assert resp.status_code == 200
  data = resp.json()

  assert data["event_id"] == 300000
  assert "features" in data
  assert "explanation" in data
  expl = data["explanation"]
  assert "attributions" in expl
  assert "object_groups" in expl


def test_permalink_holdout_404(gallery_client):
  """Assert holdout EventId 300045 and non-existent ID 999999 return 404 Not Found on permalink endpoint."""
  resp_holdout = gallery_client.get("/api/v1/events/300045/permalink")
  assert resp_holdout.status_code == 404

  resp_missing = gallery_client.get("/api/v1/events/999999/permalink")
  assert resp_missing.status_code == 404


def test_gallery_interesting_fallback_metadata(gallery_client):
  """Amendment 3: Exposes selection_method metadata ('threshold_window' or 'nearest_threshold_fallback')."""
  resp = gallery_client.get("/api/v1/events/gallery")
  assert resp.status_code == 200
  data = resp.json()

  assert "selection_method" in data
  assert data["selection_method"] in ("threshold_window", "nearest_threshold_fallback")


def test_gallery_and_permalink_p50_latency(gallery_client):
  """Amendment 5: Measured warm p50 latency < 200ms for both /events/gallery and /events/{id}/permalink."""
  # 1. Gallery p50
  gallery_client.get("/api/v1/events/gallery")  # warm up
  gallery_timings = []
  for _ in range(10):
    t0 = time.time()
    res = gallery_client.get("/api/v1/events/gallery")
    t1 = time.time()
    assert res.status_code == 200
    gallery_timings.append((t1 - t0) * 1000.0)

  p50_gallery = sorted(gallery_timings)[len(gallery_timings) // 2]
  print(f"\nMeasured /api/v1/events/gallery p50 latency: {p50_gallery:.2f} ms")
  assert p50_gallery < 200.0

  # 2. Permalink p50
  gallery_client.get("/api/v1/events/300000/permalink?model_id=xgboost")  # warm up
  permalink_timings = []
  for _ in range(10):
    t0 = time.time()
    res = gallery_client.get("/api/v1/events/300000/permalink?model_id=xgboost")
    t1 = time.time()
    assert res.status_code == 200
    permalink_timings.append((t1 - t0) * 1000.0)

  p50_permalink = sorted(permalink_timings)[len(permalink_timings) // 2]
  print(f"Measured /api/v1/events/300000/permalink p50 latency: {p50_permalink:.2f} ms")
  assert p50_permalink < 200.0
