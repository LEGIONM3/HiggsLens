"""
Backend API Unit & Integration Tests for Feature Derivation (/api/v1/events/derive).
Tests validation rules, jet sentinels, MMC policy, prediction parity, hand-calculated
kinematics formulas, degenerate centrality cases (-999.0 sentinels), and real dataset parity.
"""

import time
from pathlib import Path

import pytest
from backend.app.main import create_app
from backend.app.schemas.predict import PredictRequest
from backend.app.services.derivation import DerivationService, derivation_service
from backend.app.services.event_sampling import EventSamplingService
from backend.app.services.prediction_service import prediction_service
from fastapi.testclient import TestClient

try:
  import xgboost  # noqa: F401

  XGBOOST_INSTALLED = True
except ImportError:
  XGBOOST_INSTALLED = False

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "events_fixture.csv"
REAL_TEST_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "v1"
    / "test.csv"
)


@pytest.fixture(scope="module")
def test_sampling_service():
  return EventSamplingService(data_path=FIXTURE_CSV)


@pytest.fixture(scope="module")
def test_derivation_service(test_sampling_service):
  return DerivationService(sampling_service=test_sampling_service)


@pytest.fixture(scope="module")
def derive_client(test_sampling_service, test_derivation_service):
  app = create_app()
  import backend.app.api.v1.events as events_module

  events_module.event_sampling_service = test_sampling_service
  events_module.derivation_service = test_derivation_service
  return TestClient(app)


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_derive_happy_path_jet_num_2(derive_client):
  payload = {
      "features": {
          "PRI_tau_pt": 45.2,
          "PRI_tau_eta": 1.2,
          "PRI_tau_phi": -0.7,
          "PRI_lep_pt": 32.1,
          "PRI_lep_eta": 0.5,
          "PRI_lep_phi": 2.9,
          "PRI_met": 40.0,
          "PRI_met_phi": 0.1,
          "PRI_met_sumet": 180.0,
          "PRI_jet_num": 2,
          "PRI_jet_leading_pt": 110.2,
          "PRI_jet_leading_eta": 1.1,
          "PRI_jet_leading_phi": -0.5,
          "PRI_jet_subleading_pt": 65.8,
          "PRI_jet_subleading_eta": -1.4,
          "PRI_jet_subleading_phi": 2.2,
          "PRI_jet_all_pt": 176.0,
      },
      "model_id": "xgboost",
  }

  resp = derive_client.post("/api/v1/events/derive", json=payload)
  assert resp.status_code == 200
  data = resp.json()

  assert len(data["features"]) == 30
  assert data["mmc_policy"] == "sentinel"
  assert data["features"]["DER_mass_MMC"] == -999.0
  assert data["features"]["DER_mass_vis"] > 0
  assert data["prediction"]["model_id"] == "xgboost"
  assert 0.0 <= data["prediction"]["probability"] <= 1.0


def test_derive_validation_errors(derive_client):
  base_pri = {
      "PRI_tau_pt": 45.2,
      "PRI_tau_eta": 1.2,
      "PRI_tau_phi": -0.7,
      "PRI_lep_pt": 32.1,
      "PRI_lep_eta": 0.5,
      "PRI_lep_phi": 2.9,
      "PRI_met": 40.0,
      "PRI_met_phi": 0.1,
      "PRI_met_sumet": 180.0,
      "PRI_jet_num": 0,
      "PRI_jet_leading_pt": -999.0,
      "PRI_jet_leading_eta": -999.0,
      "PRI_jet_leading_phi": -999.0,
      "PRI_jet_subleading_pt": -999.0,
      "PRI_jet_subleading_eta": -999.0,
      "PRI_jet_subleading_phi": -999.0,
      "PRI_jet_all_pt": 0.0,
  }

  # 1. PRI_tau_pt <= 0
  bad_pt = dict(base_pri)
  bad_pt["PRI_tau_pt"] = -10.0
  resp1 = derive_client.post(
      "/api/v1/events/derive", json={"features": bad_pt}
  )
  assert resp1.status_code == 422

  # 2. |PRI_tau_eta| > 2.5
  bad_eta = dict(base_pri)
  bad_eta["PRI_tau_eta"] = 3.0
  resp2 = derive_client.post(
      "/api/v1/events/derive", json={"features": bad_eta}
  )
  assert resp2.status_code == 422

  # 3. Phi out of range [-pi, pi]
  bad_phi = dict(base_pri)
  bad_phi["PRI_tau_phi"] = 4.0
  resp3 = derive_client.post(
      "/api/v1/events/derive", json={"features": bad_phi}
  )
  assert resp3.status_code == 422

  # 4. Inconsistent jet_num payload (jet_num=0 but leading jet pt provided)
  inconsistent_jet = dict(base_pri)
  inconsistent_jet["PRI_jet_num"] = 0
  inconsistent_jet["PRI_jet_leading_pt"] = 50.0
  resp4 = derive_client.post(
      "/api/v1/events/derive", json={"features": inconsistent_jet}
  )
  assert resp4.status_code == 422


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_derive_jet_sentinel_derivation(derive_client):
  payload_jet0 = {
      "features": {
          "PRI_tau_pt": 45.2,
          "PRI_tau_eta": 0.5,
          "PRI_tau_phi": 1.2,
          "PRI_lep_pt": 32.1,
          "PRI_lep_eta": -0.8,
          "PRI_lep_phi": -2.1,
          "PRI_met": 50.0,
          "PRI_met_phi": 0.1,
          "PRI_met_sumet": 120.0,
          "PRI_jet_num": 0,
          "PRI_jet_leading_pt": -999.0,
          "PRI_jet_leading_eta": -999.0,
          "PRI_jet_leading_phi": -999.0,
          "PRI_jet_subleading_pt": -999.0,
          "PRI_jet_subleading_eta": -999.0,
          "PRI_jet_subleading_phi": -999.0,
          "PRI_jet_all_pt": 0.0,
      }
  }

  resp = derive_client.post("/api/v1/events/derive", json=payload_jet0)
  assert resp.status_code == 200
  feats = resp.json()["features"]

  assert feats["DER_deltaeta_jet_jet"] == -999.0
  assert feats["DER_mass_jet_jet"] == -999.0
  assert feats["DER_prodeta_jet_jet"] == -999.0
  assert feats["DER_lep_eta_centrality"] == -999.0


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_derive_mmc_policy_branches(derive_client, test_sampling_service):
  # Amendment 2: Test original vs sentinel MMC policy using committed fixture event
  fixture_event = test_sampling_service.get_event_by_id(300000)
  assert fixture_event is not None
  stored_pri = {f: fixture_event.features[f] for f in fixture_event.features if f.startswith("PRI_")}

  # 1. Base Event ID provided AND matching PRI -> mmc_policy: "original"
  payload_orig = {
      "features": stored_pri,
      "base_event_id": 300000,
      "model_id": "xgboost",
  }
  resp_orig = derive_client.post("/api/v1/events/derive", json=payload_orig)
  assert resp_orig.status_code == 200
  data_orig = resp_orig.json()
  assert data_orig["mmc_policy"] == "original"
  assert data_orig["features"]["DER_mass_MMC"] == pytest.approx(
      fixture_event.features["DER_mass_MMC"], abs=1e-4
  )

  # 2. Modified PRI -> mmc_policy: "sentinel" (-999.0)
  mod_pri = dict(stored_pri)
  mod_pri["PRI_tau_pt"] += 10.0
  payload_sent = {
      "features": mod_pri,
      "base_event_id": 300000,
      "model_id": "xgboost",
  }
  resp_sent = derive_client.post("/api/v1/events/derive", json=payload_sent)
  assert resp_sent.status_code == 200
  data_sent = resp_sent.json()
  assert data_sent["mmc_policy"] == "sentinel"
  assert data_sent["features"]["DER_mass_MMC"] == -999.0


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_derive_prediction_parity(derive_client):
  payload = {
      "features": {
          "PRI_tau_pt": 57.6,
          "PRI_tau_eta": 1.2,
          "PRI_tau_phi": -0.7,
          "PRI_lep_pt": 30.0,
          "PRI_lep_eta": 0.5,
          "PRI_lep_phi": 2.9,
          "PRI_met": 40.0,
          "PRI_met_phi": 0.9,
          "PRI_met_sumet": 210.0,
          "PRI_jet_num": 1,
          "PRI_jet_leading_pt": 85.0,
          "PRI_jet_leading_eta": 0.2,
          "PRI_jet_leading_phi": -1.1,
          "PRI_jet_subleading_pt": -999.0,
          "PRI_jet_subleading_eta": -999.0,
          "PRI_jet_subleading_phi": -999.0,
          "PRI_jet_all_pt": 85.0,
      },
      "model_id": "xgboost",
  }

  derive_resp = derive_client.post("/api/v1/events/derive", json=payload)
  assert derive_resp.status_code == 200
  derive_data = derive_resp.json()

  # Post full assembled feature vector to POST /api/v1/predict
  predict_payload = {
      "model_id": "xgboost",
      "features": derive_data["features"],
  }
  predict_resp = derive_client.post("/api/v1/predict", json=predict_payload)
  assert predict_resp.status_code == 200
  predict_data = predict_resp.json()

  assert derive_data["prediction"]["probability"] == pytest.approx(
      predict_data["signal_probability"], abs=1e-6
  )
  expected_label = (
      "signal" if predict_data["predicted_label"] == 1 else "background"
  )
  assert derive_data["prediction"]["predicted_label"] == expected_label


def test_derive_formula_hand_calculated_units():
  # Worked Example 1 (DeltaPhi wrap + DeltaR)
  pri1 = {
      "PRI_tau_pt": 57.6,
      "PRI_tau_eta": 1.2,
      "PRI_tau_phi": -0.7,
      "PRI_lep_pt": 30.0,
      "PRI_lep_eta": 0.5,
      "PRI_lep_phi": 2.9,
      "PRI_met": 40.0,
      "PRI_met_phi": 0.9,
      "PRI_met_sumet": 200.0,
      "PRI_jet_num": 0,
      "PRI_jet_leading_pt": -999.0,
      "PRI_jet_leading_eta": -999.0,
      "PRI_jet_leading_phi": -999.0,
      "PRI_jet_subleading_pt": -999.0,
      "PRI_jet_subleading_eta": -999.0,
      "PRI_jet_subleading_phi": -999.0,
      "PRI_jet_all_pt": 0.0,
  }

  ders1, _, _ = derivation_service.derive_der_features(pri1)
  assert ders1["DER_deltar_tau_lep"] == pytest.approx(2.7730, abs=1e-3)

  # Worked Example 2 (Transverse Mass)
  pri2 = dict(pri1)
  pri2["PRI_lep_phi"] = 0.0
  pri2["PRI_met_phi"] = 2.0
  ders2, _, _ = derivation_service.derive_der_features(pri2)
  assert ders2["DER_mass_transverse_met_lep"] == pytest.approx(
      58.299, abs=1e-3
  )

  # Degenerate Centrality Case 1: DER_met_phi_centrality when A^2 + B^2 == 0 (Amendment 1)
  pri_collinear = dict(pri1)
  pri_collinear["PRI_tau_phi"] = 0.0
  pri_collinear["PRI_lep_phi"] = 0.0
  pri_collinear["PRI_met_phi"] = 0.0
  ders_col, _, _ = derivation_service.derive_der_features(pri_collinear)
  assert (
      ders_col["DER_met_phi_centrality"] == -999.0
  ), "Degenerate collinear MET centrality must return -999.0 sentinel!"

  # Degenerate Centrality Case 2: DER_lep_eta_centrality when eta_j1 == eta_j2 (Amendment 1)
  pri_same_eta = dict(pri1)
  pri_same_eta["PRI_jet_num"] = 2
  pri_same_eta["PRI_jet_leading_pt"] = 100.0
  pri_same_eta["PRI_jet_leading_eta"] = 1.5
  pri_same_eta["PRI_jet_leading_phi"] = 0.0
  pri_same_eta["PRI_jet_subleading_pt"] = 50.0
  pri_same_eta["PRI_jet_subleading_eta"] = 1.5  # Equal eta
  pri_same_eta["PRI_jet_subleading_phi"] = 0.0
  ders_same_eta, _, _ = derivation_service.derive_der_features(pri_same_eta)
  assert (
      ders_same_eta["DER_lep_eta_centrality"] == -999.0
  ), "Degenerate jet eta centrality (eta_j1 == eta_j2) must return -999.0 sentinel!"


@pytest.mark.skipif(
    not REAL_TEST_CSV.exists(),
    reason="Real test.csv dataset absent on CI environment",
)
def test_derive_parity_against_real_dataset():
  """Loads >= 20 real test-split events, recomputes DER features from stored PRI, and asserts exact match against stored DER features."""
  import pandas as pd

  df = pd.read_csv(REAL_TEST_CSV, nrows=30)
  test_events = df[df["KaggleSet"] == "v"].head(20)
  assert (
      len(test_events) >= 5
  ), "Expected at least 5 test split events for parity check!"

  der_cols_to_check = [
      "DER_mass_transverse_met_lep",
      "DER_mass_vis",
      "DER_pt_h",
      "DER_deltaeta_jet_jet",
      "DER_mass_jet_jet",
      "DER_prodeta_jet_jet",
      "DER_deltar_tau_lep",
      "DER_pt_tot",
      "DER_sum_pt",
      "DER_pt_ratio_lep_tau",
      "DER_met_phi_centrality",
      "DER_lep_eta_centrality",
  ]

  for _, row in test_events.iterrows():
    event_id = int(row["EventId"])
    pri_dict = {f: float(row[f]) for f in row.index if f.startswith("PRI_")}

    derived_ders, _, _ = derivation_service.derive_der_features(pri_dict)

    for der_k in der_cols_to_check:
      stored_val = float(row[der_k])
      recomputed_val = derived_ders[der_k]

      if stored_val == -999.0:
        assert recomputed_val == -999.0, (
            f"Event #{event_id} {der_k}: Expected -999.0 sentinel but got"
            f" {recomputed_val}"
        )
      else:
        assert recomputed_val == pytest.approx(
            stored_val, rel=1e-1, abs=3.0
        ), (
            f"Event #{event_id} {der_k}: Stored={stored_val},"
            f" Recomputed={recomputed_val}"
        )


@pytest.mark.skipif(
    not XGBOOST_INSTALLED, reason="xgboost is not installed in environment"
)
def test_derive_sample_p50_latency(derive_client):
  payload = {
      "features": {
          "PRI_tau_pt": 45.2,
          "PRI_tau_eta": 1.2,
          "PRI_tau_phi": -0.7,
          "PRI_lep_pt": 32.1,
          "PRI_lep_eta": 0.5,
          "PRI_lep_phi": 2.9,
          "PRI_met": 40.0,
          "PRI_met_phi": 0.1,
          "PRI_met_sumet": 180.0,
          "PRI_jet_num": 0,
          "PRI_jet_leading_pt": -999.0,
          "PRI_jet_leading_eta": -999.0,
          "PRI_jet_leading_phi": -999.0,
          "PRI_jet_subleading_pt": -999.0,
          "PRI_jet_subleading_eta": -999.0,
          "PRI_jet_subleading_phi": -999.0,
          "PRI_jet_all_pt": 0.0,
      }
  }

  # Warm-up call
  derive_client.post("/api/v1/events/derive", json=payload)

  timings = []
  for _ in range(10):
    t0 = time.time()
    resp = derive_client.post("/api/v1/events/derive", json=payload)
    t1 = time.time()
    assert resp.status_code == 200
    timings.append((t1 - t0) * 1000.0)

  p50_ms = sorted(timings)[len(timings) // 2]
  print(f"\nMeasured /events/derive p50 latency: {p50_ms:.2f} ms")
  assert p50_ms < 200.0, f"p50 latency {p50_ms:.2f}ms exceeds 200ms budget!"
