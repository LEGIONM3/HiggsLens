from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from backend.app.core.config import settings
from backend.app.data.pipeline import DatasetPipeline
from backend.app.models.registry import model_registry
from backend.app.schemas.dataset import DERIVED_FEATURES, PRIMARY_FEATURES
from backend.app.schemas.predictions import (
    ComparePredictionsRequest,
    ComparePredictionsResponse,
    DetectorViewMetadata,
    MissingTransverseEnergy,
    PredictionRequest,
    PredictionResponse,
    PredictionSummary,
    ReconstructedObject,
)
from fastapi import APIRouter, HTTPException

router = APIRouter()


def _get_event_features(event_id: Optional[int], features: Optional[Dict[str, float]]) -> Tuple[pd.DataFrame, List[str]]:
    missing_adjusted: List[str] = []

    if event_id is not None:
        raw_path = settings.get_raw_dataset_path()
        if not raw_path.exists():
            raise HTTPException(status_code=400, detail="CERN dataset not found.")
        df = pd.read_csv(raw_path)
        event_df = df[df["EventId"] == event_id]
        if event_df.empty:
            raise HTTPException(status_code=404, detail=f"EventId {event_id} not found in dataset.")
        return event_df.copy(), missing_adjusted

    elif features is not None:
        # Create single row DataFrame
        row: Dict[str, Any] = {}
        for col in PRIMARY_FEATURES + DERIVED_FEATURES:
            if col in features:
                row[col] = features[col]
            else:
                row[col] = -999.0
                missing_adjusted.append(col)
        return pd.DataFrame([row]), missing_adjusted
    else:
        # If neither provided, sample the first validation event from dataset if available, or mock defaults
        raw_path = settings.get_raw_dataset_path()
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            if "KaggleSet" in df.columns:
                val_df = df[df["KaggleSet"] == "b"]
                if not val_df.empty:
                    return val_df.iloc[[0]].copy(), missing_adjusted
            return df.iloc[[0]].copy(), missing_adjusted

        # Fallback default features
        row = {c: -999.0 for c in PRIMARY_FEATURES + DERIVED_FEATURES}
        row["PRI_tau_pt"] = 45.2
        row["PRI_tau_eta"] = 0.35
        row["PRI_tau_phi"] = -1.2
        row["PRI_lep_pt"] = 52.8
        row["PRI_lep_eta"] = -0.15
        row["PRI_lep_phi"] = 2.0
        row["PRI_met"] = 65.0
        row["PRI_met_phi"] = 0.5
        row["PRI_jet_num"] = 1.0
        row["PRI_jet_leading_pt"] = 85.0
        row["PRI_jet_leading_eta"] = 1.1
        row["PRI_jet_leading_phi"] = -2.5
        return pd.DataFrame([row]), ["used_default_sample"]


def _build_visualization_payload(row_dict: Dict[str, Any]) -> Tuple[List[ReconstructedObject], MissingTransverseEnergy, Dict[str, Any]]:
    objects: List[ReconstructedObject] = []

    # Hadronic tau
    tau_pt = float(row_dict.get("PRI_tau_pt", -999.0))
    objects.append(ReconstructedObject(
        object_type="hadronic_tau",
        label="Hadronic Tau",
        pt=tau_pt,
        eta=float(row_dict.get("PRI_tau_eta", 0.0)) if tau_pt != -999.0 else 0.0,
        phi=float(row_dict.get("PRI_tau_phi", 0.0)) if tau_pt != -999.0 else 0.0,
        present=(tau_pt != -999.0 and tau_pt > 0)
    ))

    # Lepton
    lep_pt = float(row_dict.get("PRI_lep_pt", -999.0))
    objects.append(ReconstructedObject(
        object_type="lepton",
        label="Light Lepton (e/μ)",
        pt=lep_pt,
        eta=float(row_dict.get("PRI_lep_eta", 0.0)) if lep_pt != -999.0 else 0.0,
        phi=float(row_dict.get("PRI_lep_phi", 0.0)) if lep_pt != -999.0 else 0.0,
        present=(lep_pt != -999.0 and lep_pt > 0)
    ))

    # Jet count & leading/subleading
    jet_num = int(row_dict.get("PRI_jet_num", 0))
    if jet_num < 0:
        jet_num = 0

    l_pt = float(row_dict.get("PRI_jet_leading_pt", -999.0))
    objects.append(ReconstructedObject(
        object_type="leading_jet",
        label="Leading Jet",
        pt=l_pt,
        eta=float(row_dict.get("PRI_jet_leading_eta", 0.0)) if l_pt != -999.0 else 0.0,
        phi=float(row_dict.get("PRI_jet_leading_phi", 0.0)) if l_pt != -999.0 else 0.0,
        present=(jet_num >= 1 and l_pt != -999.0 and l_pt > 0)
    ))

    sl_pt = float(row_dict.get("PRI_jet_subleading_pt", -999.0))
    objects.append(ReconstructedObject(
        object_type="subleading_jet",
        label="Subleading Jet",
        pt=sl_pt,
        eta=float(row_dict.get("PRI_jet_subleading_eta", 0.0)) if sl_pt != -999.0 else 0.0,
        phi=float(row_dict.get("PRI_jet_subleading_phi", 0.0)) if sl_pt != -999.0 else 0.0,
        present=(jet_num >= 2 and sl_pt != -999.0 and sl_pt > 0)
    ))

    # MET
    met_val = float(row_dict.get("PRI_met", 0.0))
    if met_val == -999.0:
        met_val = 0.0
    sumet_raw = row_dict.get("PRI_met_sumet", -999.0)
    met = MissingTransverseEnergy(
        magnitude=met_val,
        phi=float(row_dict.get("PRI_met_phi", 0.0)) if met_val > 0 else 0.0,
        sumet=float(sumet_raw) if sumet_raw is not None and sumet_raw != -999.0 else None
    )

    # Jet summary
    tot_pt = float(row_dict.get("PRI_jet_all_pt", 0.0))
    if tot_pt == -999.0:
        tot_pt = 0.0
    jet_summary = {"count": jet_num, "total_pt": tot_pt}

    return objects, met, jet_summary


@router.post("", response_model=PredictionResponse)
def run_prediction(req: PredictionRequest):
    df_event, missing_adj = _get_event_features(req.event_id, req.features)
    row_dict = df_event.iloc[0].to_dict()

    objects, met, jet_summary = _build_visualization_payload(row_dict)

    # Get model candidate
    cand = model_registry.get_candidate(req.model_id)
    if not cand.is_fitted:
        # Check if we can fit on fast mode sample on the fly if not fitted
        raw_path = settings.get_raw_dataset_path()
        if raw_path.exists():
            pipeline = DatasetPipeline()
            parts = pipeline.get_partition_splits(sample_size=2000)
            cand.fit(parts["train"], DatasetPipeline.encode_labels(parts["train"]), feature_set=req.feature_set)
        else:
            raise HTTPException(status_code=400, detail=f"Model '{req.model_id}' is not trained yet and dataset is not available to auto-fit.")

    probs = cand.predict_proba(df_event)[0]
    sig_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    bg_prob = float(probs[0]) if len(probs) > 1 else 1.0 - sig_prob
    thresh = 0.5
    pred_class = "signal" if sig_prob >= thresh else "background"

    # Check if event is validation or test set
    val_status = "unpartitioned_event"
    if "KaggleSet" in df_event.columns:
        ks = str(df_event.iloc[0]["KaggleSet"])
        val_status = {"t": "training_set", "b": "validation_set", "v": "test_set", "u": "holdout_set"}.get(ks, "unpartitioned")

    pred_summary = PredictionSummary(
        model_id=req.model_id,
        model_version=settings.APP_VERSION,
        feature_set=req.feature_set,
        signal_probability=sig_prob,
        background_probability=bg_prob,
        predicted_class=pred_class,
        decision_threshold=thresh,
        distance_from_threshold=abs(sig_prob - thresh),
        validation_status=val_status,
        explanation_status="SHAP / explanation plugin not installed or inactive"
    )

    return PredictionResponse(
        event_id=int(df_event.iloc[0]["EventId"]) if "EventId" in df_event.columns else req.event_id,
        objects=objects,
        missing_transverse_energy=met,
        jet_summary=jet_summary,
        detector_view=DetectorViewMetadata(),
        prediction=pred_summary,
        missing_adjusted_fields=missing_adj
    )


@router.post("/compare", response_model=ComparePredictionsResponse)
def compare_predictions(req: ComparePredictionsRequest):
    df_event, _ = _get_event_features(req.event_id, req.features)
    row_dict = df_event.iloc[0].to_dict()

    objects, met, _ = _build_visualization_payload(row_dict)

    predictions: Dict[str, PredictionSummary] = {}

    val_status = "unpartitioned_event"
    if "KaggleSet" in df_event.columns:
        ks = str(df_event.iloc[0]["KaggleSet"])
        val_status = {"t": "training_set", "b": "validation_set", "v": "test_set", "u": "holdout_set"}.get(ks, "unpartitioned")

    for m_id in req.model_ids:
        try:
            cand = model_registry.get_candidate(m_id)
            if not cand.is_fitted:
                raw_path = settings.get_raw_dataset_path()
                if raw_path.exists():
                    pipeline = DatasetPipeline()
                    parts = pipeline.get_partition_splits(sample_size=2000)
                    cand.fit(parts["train"], DatasetPipeline.encode_labels(parts["train"]), feature_set=req.feature_set)
            if cand.is_fitted:
                probs = cand.predict_proba(df_event)[0]
                sig_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                bg_prob = float(probs[0]) if len(probs) > 1 else 1.0 - sig_prob
                thresh = 0.5
                pred_class = "signal" if sig_prob >= thresh else "background"
                predictions[m_id] = PredictionSummary(
                    model_id=m_id,
                    model_version=settings.APP_VERSION,
                    feature_set=req.feature_set,
                    signal_probability=sig_prob,
                    background_probability=bg_prob,
                    predicted_class=pred_class,
                    decision_threshold=thresh,
                    distance_from_threshold=abs(sig_prob - thresh),
                    validation_status=val_status
                )
        except Exception:
            continue

    return ComparePredictionsResponse(
        event_id=int(df_event.iloc[0]["EventId"]) if "EventId" in df_event.columns else req.event_id,
        objects=objects,
        missing_transverse_energy=met,
        detector_view=DetectorViewMetadata(),
        predictions=predictions
    )
