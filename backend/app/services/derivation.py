"""
Feature Derivation Service for HiggsLens (/api/v1/events/derive).
Re-computes the 13 DER_* features from 17 raw PRI_* inputs using massless 4-momentum kinematics,
implements MMC sentinel policy, and routes inference through certified PredictionService.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from backend.app.schemas.derive import DeriveRequest, DeriveResponse
from backend.app.schemas.events import EventPredictionResponse
from backend.app.schemas.predict import PredictRequest
from backend.app.services.event_sampling import (
    EventSamplingService,
    event_sampling_service,
)
from backend.app.services.prediction_service import (
    PredictionService,
    prediction_service,
)
from ml.data.feature_sets import PRIMARY_FEATURES

logger = logging.getLogger("higgslens.derivation")


def wrap_angle(angle: float) -> float:
  """Wraps an angle into the [-pi, pi] interval."""
  return float(((angle + math.pi) % (2 * math.pi)) - math.pi)


class DerivationService:
  """Service for validating primary physics features and re-deriving calculated DER_* features."""

  def __init__(
      self,
      sampling_service: Optional[EventSamplingService] = None,
      pred_service: Optional[PredictionService] = None,
  ):
    self.sampling_service = sampling_service or event_sampling_service
    self.pred_service = pred_service or prediction_service

  def validate_pri_features(self, features: Dict[str, float]) -> None:
    """Validates physical bounds and sentinel consistency for all 17 PRI_* features.

    Raises ValueError on any violation (caught in router as 422).
    """
    missing = [f for f in PRIMARY_FEATURES if f not in features]
    if missing:
      raise ValueError(f"Missing required primary feature(s): {missing}")

    tau_pt = features["PRI_tau_pt"]
    lep_pt = features["PRI_lep_pt"]
    met = features["PRI_met"]
    tau_eta = features["PRI_tau_eta"]
    lep_eta = features["PRI_lep_eta"]
    jet_num = int(round(features["PRI_jet_num"]))

    # Physical bounds checks
    if tau_pt <= 0:
      raise ValueError(f"PRI_tau_pt must be > 0 (got {tau_pt})")
    if lep_pt <= 0:
      raise ValueError(f"PRI_lep_pt must be > 0 (got {lep_pt})")
    if met < 0:
      raise ValueError(f"PRI_met must be >= 0 (got {met})")
    if abs(tau_eta) > 2.5:
      raise ValueError(f"|PRI_tau_eta| must be <= 2.5 (got {tau_eta})")
    if abs(lep_eta) > 2.5:
      raise ValueError(f"|PRI_lep_eta| must be <= 2.5 (got {lep_eta})")

    # Phi bounds check [-pi, pi]
    phi_keys = [k for k in PRIMARY_FEATURES if "phi" in k]
    for pk in phi_keys:
      val = features[pk]
      if val != -999.0 and (val < -math.pi - 1e-4 or val > math.pi + 1e-4):
        raise ValueError(f"Feature '{pk}' must be in interval [-pi, pi] (got {val})")

    # Jet multiplicity check
    if jet_num not in (0, 1, 2, 3):
      raise ValueError(
          f"PRI_jet_num must be 0, 1, 2, or 3 (got {jet_num})"
      )

    # Jet-dependent sentinel consistency
    if jet_num == 0:
      lead_sentinels = [
          'PRI_jet_leading_pt',
          'PRI_jet_leading_eta',
          'PRI_jet_leading_phi',
      ]
      for k in lead_sentinels:
        if features[k] != -999.0:
          raise ValueError(
              f"Inconsistent jet payload: PRI_jet_num is 0 but '{k}' is not"
              f' -999.0 sentinel (got {features[k]})'
          )
    elif jet_num >= 1:
      if features['PRI_jet_leading_pt'] == -999.0:
        raise ValueError(
            'Inconsistent jet payload: PRI_jet_num >= 1 but PRI_jet_leading_pt'
            ' is -999.0'
        )
      if abs(features['PRI_jet_leading_eta']) > 4.5:
        raise ValueError(
            '|PRI_jet_leading_eta| must be <= 4.5 (got'
            f" {features['PRI_jet_leading_eta']})"
        )

    if jet_num <= 1:
      sublead_sentinels = [
          'PRI_jet_subleading_pt',
          'PRI_jet_subleading_eta',
          'PRI_jet_subleading_phi',
      ]
      for k in sublead_sentinels:
        if features[k] != -999.0:
          raise ValueError(
              f"Inconsistent jet payload: PRI_jet_num is {jet_num} but '{k}' is"
              f' not -999.0 sentinel (got {features[k]})'
          )
    elif jet_num >= 2:
      if features['PRI_jet_subleading_pt'] == -999.0:
        raise ValueError(
            'Inconsistent jet payload: PRI_jet_num >= 2 but'
            ' PRI_jet_subleading_pt is -999.0'
        )
      if abs(features['PRI_jet_subleading_eta']) > 4.5:
        raise ValueError(
            '|PRI_jet_subleading_eta| must be <= 4.5 (got'
            f" {features['PRI_jet_subleading_eta']})"
        )

  def derive_der_features(
      self, pri: Dict[str, float], base_event_id: Optional[int] = None
  ) -> Tuple[Dict[str, float], str, List[str]]:
    """Recomputes all 13 DER_* features from 17 PRI_* inputs."""
    notes: List[str] = []

    # Extract PRI inputs
    tau_pt = pri['PRI_tau_pt']
    tau_eta = pri['PRI_tau_eta']
    tau_phi = pri['PRI_tau_phi']

    lep_pt = pri['PRI_lep_pt']
    lep_eta = pri['PRI_lep_eta']
    lep_phi = pri['PRI_lep_phi']

    met = pri['PRI_met']
    met_phi = pri['PRI_met_phi']

    jet_num = int(round(pri['PRI_jet_num']))
    jet_all_pt = pri['PRI_jet_all_pt']

    j1_pt = pri['PRI_jet_leading_pt']
    j1_eta = pri['PRI_jet_leading_eta']
    j1_phi = pri['PRI_jet_leading_phi']

    j2_pt = pri['PRI_jet_subleading_pt']
    j2_eta = pri['PRI_jet_subleading_eta']
    j2_phi = pri['PRI_jet_subleading_phi']

    # Massless 4-momenta for tau and lep
    E_tau = tau_pt * math.cosh(tau_eta)
    px_tau = tau_pt * math.cos(tau_phi)
    py_tau = tau_pt * math.sin(tau_phi)
    pz_tau = tau_pt * math.sinh(tau_eta)

    E_lep = lep_pt * math.cosh(lep_eta)
    px_lep = lep_pt * math.cos(lep_phi)
    py_lep = lep_pt * math.sin(lep_phi)
    pz_lep = lep_pt * math.sinh(lep_eta)

    px_met = met * math.cos(met_phi)
    py_met = met * math.sin(met_phi)

    # 1. DER_mass_transverse_met_lep
    dphi_lep_met = wrap_angle(lep_phi - met_phi)
    der_mass_transverse_met_lep = math.sqrt(
        max(0.0, 2.0 * lep_pt * met * (1.0 - math.cos(dphi_lep_met)))
    )

    # 2. DER_mass_vis (tau + lep invariant mass)
    E_vis = E_tau + E_lep
    px_vis = px_tau + px_lep
    py_vis = py_tau + py_lep
    pz_vis = pz_tau + pz_lep
    m2_vis = E_vis**2 - (px_vis**2 + py_vis**2 + pz_vis**2)
    der_mass_vis = math.sqrt(max(0.0, m2_vis))

    # 3. DER_pt_h (|pT_tau + pT_lep + pT_met|)
    px_h = px_tau + px_lep + px_met
    py_h = py_tau + py_lep + py_met
    der_pt_h = math.hypot(px_h, py_h)

    # 4, 5, 6. Di-jet quantities (jet_num >= 2)
    if jet_num >= 2 and j1_pt != -999.0 and j2_pt != -999.0:
      der_deltaeta_jet_jet = abs(j1_eta - j2_eta)

      E_j1 = j1_pt * math.cosh(j1_eta)
      px_j1 = j1_pt * math.cos(j1_phi)
      py_j1 = j1_pt * math.sin(j1_phi)
      pz_j1 = j1_pt * math.sinh(j1_eta)

      E_j2 = j2_pt * math.cosh(j2_eta)
      px_j2 = j2_pt * math.cos(j2_phi)
      py_j2 = j2_pt * math.sin(j2_phi)
      pz_j2 = j2_pt * math.sinh(j2_eta)

      E_jj = E_j1 + E_j2
      px_jj = px_j1 + px_j2
      py_jj = py_j1 + py_j2
      pz_jj = pz_j1 + pz_j2
      m2_jj = E_jj**2 - (px_jj**2 + py_jj**2 + pz_jj**2)
      der_mass_jet_jet = math.sqrt(max(0.0, m2_jj))

      der_prodeta_jet_jet = j1_eta * j2_eta
    else:
      der_deltaeta_jet_jet = -999.0
      der_mass_jet_jet = -999.0
      der_prodeta_jet_jet = -999.0

    # 7. DER_deltar_tau_lep
    deta_tau_lep = tau_eta - lep_eta
    dphi_tau_lep = wrap_angle(tau_phi - lep_phi)
    der_deltar_tau_lep = math.hypot(deta_tau_lep, dphi_tau_lep)

    # 8. DER_pt_tot (|pT_tau + pT_lep + pT_j1 + pT_j2 + pT_met|)
    px_tot = px_tau + px_lep + px_met
    py_tot = py_tau + py_lep + py_met

    if jet_num >= 1 and j1_pt != -999.0:
      px_tot += j1_pt * math.cos(j1_phi)
      py_tot += j1_pt * math.sin(j1_phi)
    if jet_num >= 2 and j2_pt != -999.0:
      px_tot += j2_pt * math.cos(j2_phi)
      py_tot += j2_pt * math.sin(j2_phi)

    der_pt_tot = math.hypot(px_tot, py_tot)

    # 9. DER_sum_pt (scalar sum)
    der_sum_pt = tau_pt + lep_pt + jet_all_pt

    # 10. DER_pt_ratio_lep_tau
    der_pt_ratio_lep_tau = lep_pt / tau_pt

    # 11. DER_met_phi_centrality (official ATLAS angular ordering)
    dphi_lt = wrap_angle(lep_phi - tau_phi)
    if dphi_lt >= 0.0:
      phi1, phi2 = tau_phi, lep_phi
    else:
      phi1, phi2 = lep_phi, tau_phi

    A = math.sin(wrap_angle(met_phi - phi1))
    B = math.sin(wrap_angle(phi2 - met_phi))
    denom2 = A**2 + B**2
    if denom2 == 0.0:
      der_met_phi_centrality = -999.0
      notes.append(
          'DER_met_phi_centrality evaluated to degenerate collinear case (A^2'
          ' + B^2 == 0); assigned -999.0 sentinel.'
      )
    else:
      der_met_phi_centrality = (A + B) / math.sqrt(denom2)

    # 12. DER_lep_eta_centrality (Amendment 1)
    if jet_num >= 2 and j1_pt != -999.0 and j2_pt != -999.0:
      deta_jj = j1_eta - j2_eta
      if deta_jj == 0.0:
        der_lep_eta_centrality = -999.0
        notes.append(
            'DER_lep_eta_centrality evaluated to degenerate case'
            ' (eta_j1 == eta_j2); assigned -999.0 sentinel.'
        )
      else:
        eta_avg = (j1_eta + j2_eta) / 2.0
        exponent = -4.0 / (deta_jj**2) * ((lep_eta - eta_avg) ** 2)
        der_lep_eta_centrality = math.exp(exponent)
    else:
      der_lep_eta_centrality = -999.0

    # 13. DER_mass_MMC (Amendment 2: base_event_id lookup via EventSamplingService)
    mmc_policy = 'sentinel'
    der_mass_mmc = -999.0

    if base_event_id is not None:
      stored_event = self.sampling_service.get_event_by_id(base_event_id)
      if stored_event is not None:
        # Check if submitted 17 PRI features match stored event PRI features
        stored_feats = stored_event.features
        pri_match = True
        for pk in PRIMARY_FEATURES:
          if not math.isclose(pri[pk], stored_feats[pk], abs_tol=1e-4):
            pri_match = False
            break

        if pri_match:
          mmc_policy = 'original'
          der_mass_mmc = stored_feats['DER_mass_MMC']
          notes.append(
              f'DER_mass_MMC restored from base_event_id #{base_event_id}'
              ' (original MMC policy).'
          )
        else:
          notes.append(
              f'base_event_id #{base_event_id} provided, but submitted PRI'
              ' features differ from stored event. Applied MMC sentinel'
              ' policy (-999.0).'
          )
      else:
        notes.append(
            f'base_event_id #{base_event_id} not found in test split. Applied'
            ' MMC sentinel policy (-999.0).'
        )
    else:
      notes.append(
          'DER_mass_MMC cannot be recomputed from edited kinematics;'
          " the model receives 'not available' (-999.0), matching the dataset"
          ' convention.'
      )

    derived_ders = {
        'DER_mass_MMC': der_mass_mmc,
        'DER_mass_transverse_met_lep': der_mass_transverse_met_lep,
        'DER_mass_vis': der_mass_vis,
        'DER_pt_h': der_pt_h,
        'DER_deltaeta_jet_jet': der_deltaeta_jet_jet,
        'DER_mass_jet_jet': der_mass_jet_jet,
        'DER_prodeta_jet_jet': der_prodeta_jet_jet,
        'DER_deltar_tau_lep': der_deltar_tau_lep,
        'DER_pt_tot': der_pt_tot,
        'DER_sum_pt': der_sum_pt,
        'DER_pt_ratio_lep_tau': der_pt_ratio_lep_tau,
        'DER_met_phi_centrality': der_met_phi_centrality,
        'DER_lep_eta_centrality': der_lep_eta_centrality,
    }

    return derived_ders, mmc_policy, notes

  def derive_and_predict(self, request: DeriveRequest) -> DeriveResponse:
    """Validates PRI features, derives DER features, and routes through PredictionService."""
    # 1. Validate PRI inputs
    self.validate_pri_features(request.features)

    # 2. Derive DER features
    derived_ders, mmc_policy, notes = self.derive_der_features(
        request.features, request.base_event_id
    )

    # 3. Assemble full 30-feature dictionary
    full_features: Dict[str, float] = {}
    for k in request.features:
      full_features[k] = request.features[k]
    for k in derived_ders:
      full_features[k] = derived_ders[k]

    # 4. Route through certified PredictionService
    pred_req = PredictRequest(
        model_id=request.model_id, features=full_features, threshold=None
    )
    pred_resp = self.pred_service.predict(pred_req)

    pred_label_str = (
        'signal' if pred_resp.predicted_label == 1 else 'background'
    )

    return DeriveResponse(
        features=full_features,
        prediction=EventPredictionResponse(
            model_id=request.model_id,
            probability=pred_resp.signal_probability,
            predicted_label=pred_label_str,
            threshold=pred_resp.threshold_used,
        ),
        mmc_policy=mmc_policy,
        notes=notes,
    )


derivation_service = DerivationService()
