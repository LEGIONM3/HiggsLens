"""
Backend API Unit Tests for Event Sampling (/api/v1/events).
Tests sampling, single event retrieval, holdout exclusion, prediction parity,
sentinel passthrough, dynamic threshold reading, and 503 missing dataset handling.
"""

import time
from pathlib import Path

import pytest
from backend.app.main import create_app
from backend.app.schemas.predict import PredictRequest
from backend.app.services.event_sampling import EventSamplingService
from backend.app.services.prediction_service import prediction_service
from fastapi.testclient import TestClient

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "events_fixture.csv"
REAL_TEST_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "v1" / "test.csv"


@pytest.fixture(scope="module")
def test_sampling_service():
    """Returns an EventSamplingService instance pointing to the CI-safe test fixture."""
    return EventSamplingService(data_path=FIXTURE_CSV)


@pytest.fixture(scope="module")
def events_client(test_sampling_service):
    """FastAPI TestClient with event_sampling_service patched to use the fixture."""
    app = create_app()
    import backend.app.api.v1.events as events_module
    events_module.event_sampling_service = test_sampling_service
    return TestClient(app)


def test_events_sample_happy_path(events_client):
    resp = events_client.get("/api/v1/events/sample?n=12&seed=42&label=any")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 12
    assert len(data["events"]) == 12
    assert data["seed"] == 42
    assert data["label_filter"] == "any"

    first_event = data["events"][0]
    assert "event_id" in first_event
    assert first_event["true_label"] in ("signal", "background")
    assert len(first_event["features"]) == 30
    assert first_event["prediction"]["model_id"] == "xgboost"
    assert 0.0 <= first_event["prediction"]["probability"] <= 1.0
    assert first_event["prediction"]["predicted_label"] in ("signal", "background")


def test_events_sample_hard_cap_422(events_client):
    # n=51 exceeds hard cap of 50
    resp = events_client.get("/api/v1/events/sample?n=51")
    assert resp.status_code == 422

    # n=0 is invalid
    resp_zero = events_client.get("/api/v1/events/sample?n=0")
    assert resp_zero.status_code == 422


def test_events_sample_seeded_determinism(events_client):
    resp1 = events_client.get("/api/v1/events/sample?n=10&seed=123")
    resp2 = events_client.get("/api/v1/events/sample?n=10&seed=123")
    assert resp1.status_code == 200
    assert resp2.status_code == 200

    ids1 = [e["event_id"] for e in resp1.json()["events"]]
    ids2 = [e["event_id"] for e in resp2.json()["events"]]
    assert ids1 == ids2, "Seeded sampling must be 100% deterministic!"


def test_events_holdout_exclusion(events_client):
    # In fixture, EventIds 300000..300039 are test 'v', 300040..300049 are holdout 'u'
    resp = events_client.get("/api/v1/events/sample?n=40&seed=42")
    assert resp.status_code == 200
    sampled_ids = [e["event_id"] for e in resp.json()["events"]]

    # Assert no holdout IDs (300040+) are returned in sampling
    for eid in sampled_ids:
        assert eid < 300040, f"Holdout EventId {eid} was leaked into sample response!"

    # Direct GET of holdout EventId 300045 must return 404
    resp_holdout = events_client.get("/api/v1/events/300045")
    assert resp_holdout.status_code == 404
    assert "not found in test split" in resp_holdout.json()["detail"]


def test_events_prediction_correctness(events_client):
    # Amendment 2: Assert /events prediction equals /predict response for exact feature payload
    resp = events_client.get("/api/v1/events/sample?n=1&seed=42")
    assert resp.status_code == 200
    event_data = resp.json()["events"][0]

    sample_prob = event_data["prediction"]["probability"]
    sample_label = event_data["prediction"]["predicted_label"]

    # Direct call to PredictionService /predict
    predict_req = PredictRequest(
        model_id="xgboost",
        features=event_data["features"],
        threshold=None
    )
    direct_resp = prediction_service.predict(predict_req)

    assert sample_prob == pytest.approx(direct_resp.signal_probability, abs=1e-6)
    expected_direct_label = "signal" if direct_resp.predicted_label == 1 else "background"
    assert sample_label == expected_direct_label


def test_events_dynamic_threshold(events_client):
    # Amendment 3: Assert API-returned threshold equals metrics["optimal_threshold"] from artifact
    resp = events_client.get("/api/v1/events/sample?n=1&seed=42")
    assert resp.status_code == 200
    event_data = resp.json()["events"][0]

    api_thresh = event_data["prediction"]["threshold"]

    # Read artifact metrics directly
    artifact = prediction_service.registry.get_artifact("xgboost")
    expected_thresh = float(artifact.metrics["optimal_threshold"])

    assert api_thresh == pytest.approx(expected_thresh, abs=1e-6)


def test_events_sentinel_passthrough(events_client):
    resp = events_client.get("/api/v1/events/sample?n=20&seed=42")
    assert resp.status_code == 200
    events = resp.json()["events"]

    # Verify that -999.0 sentinels survive JSON serialization untouched
    found_sentinel = False
    for ev in events:
        for feat, val in ev["features"].items():
            if val == -999.0:
                found_sentinel = True
                break

    assert found_sentinel, "Expected -999.0 sentinel features in fixture events!"


def test_events_missing_dataset_503():
    # Amendment 4: Missing dataset CSV must return 503 gracefully without crashing backend startup
    non_existent_path = Path("/non_existent_directory/missing_test.csv")
    missing_service = EventSamplingService(data_path=non_existent_path)

    import backend.app.api.v1.events as events_module
    old_service = events_module.event_sampling_service
    events_module.event_sampling_service = missing_service
    try:
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/events/sample?n=5")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
    finally:
        events_module.event_sampling_service = old_service


def test_events_sample_p50_latency(events_client):
    # Amendment 5: Measure p50 latency after warm-up
    # Warm-up call
    events_client.get("/api/v1/events/sample?n=12&seed=42")

    timings = []
    for _ in range(10):
        t0 = time.time()
        resp = events_client.get("/api/v1/events/sample?n=12&seed=42")
        t1 = time.time()
        assert resp.status_code == 200
        timings.append((t1 - t0) * 1000.0)  # ms

    p50_ms = sorted(timings)[len(timings) // 2]
    print(f"\nMeasured /events/sample p50 latency: {p50_ms:.2f} ms")
    assert p50_ms < 200.0, f"p50 latency {p50_ms:.2f}ms exceeds 200ms budget!"
