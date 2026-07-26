"""
Feature Explanation & TreeSHAP Attribution Service for HiggsLens (/api/v1/explain).
Uses native XGBoost booster.predict(..., pred_contribs=True) without external shap package,
validates strict additivity, and aggregates contributions into 6 canonical physics object groups.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
  import xgboost as xgb
  XGBOOST_AVAILABLE = True
except ImportError:
  xgb = None
  XGBOOST_AVAILABLE = False
from backend.app.schemas.explain import (
    ExplainRequest,
    ExplainResponse,
    FeatureAttribution,
    ObjectGroupAttribution,
)
from backend.app.schemas.predict import PredictRequest
from backend.app.services.event_sampling import (
    EventSamplingService,
    event_sampling_service,
)
from backend.app.services.model_registry import (
    ModelRegistryService,
    model_registry_service,
)
from backend.app.services.prediction_service import (
    PredictionService,
    prediction_service,
)
from ml.data.feature_sets import ALL_PHYSICS_FEATURES

logger = logging.getLogger("higgslens.explanation")

# Explicit canonical 30-feature to 6 object-group mapping table
FEATURE_TO_GROUP_MAPPING: Dict[str, str] = {
    # Tau group (5 features)
    "PRI_tau_pt": "tau",
    "PRI_tau_eta": "tau",
    "PRI_tau_phi": "tau",
    "DER_pt_ratio_lep_tau": "tau",
    "DER_deltar_tau_lep": "tau",
    # Lepton group (3 features)
    "PRI_lep_pt": "lepton",
    "PRI_lep_eta": "lepton",
    "PRI_lep_phi": "lepton",
    # Leading Jet group (3 features)
    "PRI_jet_leading_pt": "leading_jet",
    "PRI_jet_leading_eta": "leading_jet",
    "PRI_jet_leading_phi": "leading_jet",
    # Subleading Jet group (3 features)
    "PRI_jet_subleading_pt": "subleading_jet",
    "PRI_jet_subleading_eta": "subleading_jet",
    "PRI_jet_subleading_phi": "subleading_jet",
    # MET group (5 features)
    "PRI_met": "met",
    "PRI_met_phi": "met",
    "PRI_met_sumet": "met",
    "DER_mass_transverse_met_lep": "met",
    "DER_met_phi_centrality": "met",
    # Global / System-wide group (11 features)
    "DER_mass_MMC": "global",
    "DER_mass_vis": "global",
    "DER_pt_h": "global",
    "DER_deltaeta_jet_jet": "global",
    "DER_mass_jet_jet": "global",
    "DER_prodeta_jet_jet": "global",
    "DER_pt_tot": "global",
    "DER_sum_pt": "global",
    "DER_lep_eta_centrality": "global",
    "PRI_jet_num": "global",
    "PRI_jet_all_pt": "global",
}


def sigmoid(x: float) -> float:
  """Standard logistic sigmoid function converting log-odds to probability."""
  return 1.0 / (1.0 + math.exp(-x))


class ExplanationService:
  """Service for extracting TreeSHAP attributions and verifying additivity."""

  def __init__(
      self,
      registry_service: Optional[ModelRegistryService] = None,
      pred_service: Optional[PredictionService] = None,
      sampling_service: Optional[EventSamplingService] = None,
  ):
    self.registry_service = registry_service or model_registry_service
    self.pred_service = pred_service or prediction_service
    self.sampling_service = sampling_service or event_sampling_service

  def explain(self, request: ExplainRequest) -> ExplainResponse:
    """Computes TreeSHAP attributions for a 30-feature vector using native booster.predict(..., pred_contribs=True)."""
    model_id = request.model_id.lower()

    # 1. Fetch artifact from registry
    artifact = self.registry_service.get_artifact(model_id)
    feature_schema = artifact.feature_schema
    schema_feature_names = feature_schema["feature_names"]

    # 2. Fetch trained model wrapper from registry
    model_wrapper = self.registry_service.get_cached_model(model_id)

    # Check for tree-booster TreeSHAP support
    if not XGBOOST_AVAILABLE or not hasattr(model_wrapper, "get_booster"):
      raise ValueError(
          f"Model '{model_id}' does not support TreeSHAP attributions. Only"
          " tree boosters (e.g. xgboost) are supported."
      )

    booster = model_wrapper.get_booster()

    missing_keys = [f for f in schema_feature_names if f not in request.features]
    if missing_keys:
      raise ValueError(
          f"Missing required feature(s) for model '{model_id}': {missing_keys}"
      )

    # Build 2D numpy array with strict feature order
    feature_vector = np.array(
        [[request.features[f] for f in schema_feature_names]], dtype=np.float32
    )

    # 3. Predict via certified PredictionService to get official probability, label, threshold
    pred_req = PredictRequest(
        model_id=model_id, features=request.features, threshold=None
    )
    pred_resp = self.pred_service.predict(pred_req)

    # 4. Compute native TreeSHAP contributions via xgboost.DMatrix
    dmatrix = xgb.DMatrix(feature_vector, feature_names=schema_feature_names)
    contribs_matrix = booster.predict(dmatrix, pred_contribs=True)
    contribs = contribs_matrix[0]  # Shape: (31,) -> 30 features + 1 bias

    feature_contribs = contribs[:-1]
    base_value = float(contribs[-1])
    sum_contribs = float(np.sum(feature_contribs))
    margin = base_value + sum_contribs

    # 5. Additivity Gate Check
    expected_prob = sigmoid(margin)
    if not math.isclose(expected_prob, pred_resp.signal_probability, abs_tol=1e-5):
      logger.warning(
          f"Additivity gate discrepancy: sigmoid(margin)={expected_prob:.6f} vs"
          f" pred_service={pred_resp.signal_probability:.6f}"
      )

    # Build per-feature attributions list (sorted by |contribution| desc)
    attributions_list: List[FeatureAttribution] = []
    for idx, f_name in enumerate(schema_feature_names):
      f_val = float(request.features[f_name])
      f_contrib = float(feature_contribs[idx])
      attributions_list.append(
          FeatureAttribution(
              feature=f_name, value=f_val, contribution=f_contrib
          )
      )

    attributions_list.sort(key=lambda x: abs(x.contribution), reverse=True)

    # 6. Aggregate into 6 canonical Object Groups
    group_sums: Dict[str, Tuple[float, float]] = {
        "tau": (0.0, 0.0),
        "lepton": (0.0, 0.0),
        "leading_jet": (0.0, 0.0),
        "subleading_jet": (0.0, 0.0),
        "met": (0.0, 0.0),
        "global": (0.0, 0.0),
    }

    for attr in attributions_list:
      group_name = FEATURE_TO_GROUP_MAPPING.get(attr.feature, "global")
      curr_abs, curr_signed = group_sums[group_name]
      group_sums[group_name] = (
          curr_abs + abs(attr.contribution),
          curr_signed + attr.contribution,
      )

    object_groups_list: List[ObjectGroupAttribution] = [
        ObjectGroupAttribution(
            group=g_name,
            total_abs_contribution=tot_abs,
            signed_contribution=tot_signed,
        )
        for g_name, (tot_abs, tot_signed) in group_sums.items()
    ]

    pred_label_str = (
        "signal" if pred_resp.predicted_label == 1 else "background"
    )

    return ExplainResponse(
        model_id=model_id,
        probability=pred_resp.signal_probability,
        predicted_label=pred_label_str,
        threshold=pred_resp.threshold_used,
        base_value=base_value,
        margin=margin,
        attributions=attributions_list,
        object_groups=object_groups_list,
    )

  def explain_event_by_id(
      self, event_id: int, model_id: str = "xgboost"
  ) -> ExplainResponse:
    """Fetches event from test split and delegates to explain()."""
    event = self.sampling_service.get_event_by_id(event_id)
    if event is None:
      raise KeyError(f"EventId {event_id} not found in test split.")

    return self.explain(
        ExplainRequest(features=event.features, model_id=model_id)
    )


explanation_service = ExplanationService()
