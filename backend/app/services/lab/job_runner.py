"""
Disk-Persisted Asynchronous Experiment Job Runner for HiggsLens Lab Sandboxed Zone.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.lab import LabExperimentDetailResponse, LabExperimentSummarySchema
from fastapi import HTTPException
from ml.evaluation.contract import EvaluationResult
from ml.evaluation.metrics import evaluate_threshold_scan
from ml.models.factory import build_model
from ml.models.registry import MODEL_SPECS
from sklearn.model_selection import train_test_split

logger = logging.getLogger("higgslens.lab.job_runner")


class LabJobRunner:
    """
    Asynchronous experiment job manager persisting job state and trained candidate models
    strictly to models/lab_artifacts/{experiment_id}/.
    """

    MODEL_SUBSAMPLE_CAPS: Dict[str, int] = {
        "svm_rbf": 10000,
        "quantum_kernel_svm": 1000,
        "variational_quantum_classifier": 1000,
    }

    def __init__(
        self,
        lab_data_dir: Optional[Path] = None,
        lab_artifacts_dir: Optional[Path] = None
    ):
        self.lab_data_dir = lab_data_dir or settings.LAB_DATA_DIR
        self.lab_artifacts_dir = lab_artifacts_dir or settings.LAB_ARTIFACTS_DIR
        self.lab_artifacts_dir.mkdir(parents=True, exist_ok=True)

    def check_concurrency_cap(self) -> None:
        """Enforces max concurrent experiment jobs cap (LAB_MAX_CONCURRENT_EXPERIMENTS = 1)."""
        active_jobs = []
        if not self.lab_artifacts_dir.exists():
            return

        for item in self.lab_artifacts_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "experiment_manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        status = data.get("status", "")
                        if status in ("queued", "running"):
                            active_jobs.append(data.get("experiment_id", item.name))
                    except Exception:
                        pass

        if len(active_jobs) >= settings.LAB_MAX_CONCURRENT_EXPERIMENTS:
            raise HTTPException(
                status_code=409,
                detail=f"An experiment job is currently running ({active_jobs[0]}). Please wait for it to complete."
            )

    def write_experiment_manifest(self, manifest_data: Dict[str, Any]) -> None:
        """Writes or updates experiment_manifest.json directly to disk."""
        exp_id = manifest_data["experiment_id"]
        exp_dir = self.lab_artifacts_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = exp_dir / "experiment_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def get_experiment_summary(self, experiment_id: str) -> LabExperimentSummarySchema:
        """Reads experiment summary from disk."""
        clean_id = Path(experiment_id).name
        manifest_path = self.lab_artifacts_dir / clean_id / "experiment_manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"Lab experiment '{experiment_id}' not found.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return LabExperimentSummarySchema(
            experiment_id=data["experiment_id"],
            dataset_id=data["dataset_id"],
            status=data["status"],
            model_ids=data["model_ids"],
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
        )

    def get_experiment_detail(self, experiment_id: str) -> LabExperimentDetailResponse:
        """Reads complete experiment detail and per-model leaderboard results from disk."""
        summary = self.get_experiment_summary(experiment_id)
        clean_id = Path(experiment_id).name
        exp_dir = self.lab_artifacts_dir / clean_id

        manifest_path = exp_dir / "experiment_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)

        per_model_results = exp_data.get("per_model_results", {})

        return LabExperimentDetailResponse(
            summary=summary,
            dataset_manifest=exp_data.get("dataset_manifest"),
            split_config=exp_data.get("split_config", {"train": 0.7, "validation": 0.15, "test": 0.15}),
            seed=exp_data.get("seed", 42),
            per_model_results=per_model_results,
        )

    def list_experiments(self) -> List[LabExperimentSummarySchema]:
        """Lists all recorded lab experiments from disk."""
        experiments: List[LabExperimentSummarySchema] = []
        if not self.lab_artifacts_dir.exists():
            return experiments

        for item in self.lab_artifacts_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "experiment_manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        experiments.append(
                            LabExperimentSummarySchema(
                                experiment_id=data["experiment_id"],
                                dataset_id=data["dataset_id"],
                                status=data["status"],
                                model_ids=data["model_ids"],
                                created_at=data["created_at"],
                                completed_at=data.get("completed_at"),
                                error_message=data.get("error_message"),
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read experiment manifest {manifest_path}: {e}")

        experiments.sort(key=lambda e: e.created_at, reverse=True)
        return experiments

    def create_and_enqueue_job(
        self,
        dataset_id: str,
        model_ids: List[str],
        split_config: Dict[str, float],
        seed: int = 42,
        sentinel_strategy: str = "keep-as-value"
    ) -> str:
        """
        Validates request parameters, checks concurrency cap, creates experiment manifest on disk,
        and returns server-generated experiment_id UUID.
        """
        # 1. Enforce concurrency cap
        self.check_concurrency_cap()

        # 2. Validate dataset existence
        dataset_path = self.lab_data_dir / Path(dataset_id).name / "dataset.csv"
        manifest_path = self.lab_data_dir / Path(dataset_id).name / "manifest.json"
        if not dataset_path.exists() or not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"Lab dataset '{dataset_id}' not found.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            ds_manifest = json.load(f)

        # 3. Validate model IDs
        for m_id in model_ids:
            if m_id not in MODEL_SPECS:
                raise HTTPException(status_code=404, detail=f"Unknown model_id '{m_id}'.")

        if len(model_ids) > settings.LAB_MAX_MODELS_PER_EXPERIMENT:
            raise HTTPException(
                status_code=422,
                detail=f"Requested {len(model_ids)} models exceeds maximum allowed cap of {settings.LAB_MAX_MODELS_PER_EXPERIMENT}."
            )

        # 4. Generate experiment UUID and record initial state on disk
        experiment_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        exp_manifest = {
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "status": "queued",
            "model_ids": model_ids,
            "split_config": split_config,
            "seed": seed,
            "sentinel_strategy": sentinel_strategy,
            "created_at": created_at,
            "completed_at": None,
            "error_message": None,
            "dataset_manifest": ds_manifest,
            "per_model_results": {},
        }

        self.write_experiment_manifest(exp_manifest)
        logger.info(f"Enqueued experiment job {experiment_id} for dataset {dataset_id}")
        return experiment_id

    def execute_experiment_job(self, experiment_id: str) -> None:
        """
        Executes background training job:
        1. Loads dataset and splits 70/15/15 into train/validation/test.
        2. Fits model on train (or subsampled train for expensive models).
        3. Selects optimal threshold on validation split via evaluate_threshold_scan().
        4. Reports final EvaluationResult metrics on test split at that optimal threshold.
        5. Persists artifacts under models/lab_artifacts/{experiment_id}/{model_id}/.
        6. Updates disk experiment_manifest.json with progress and status.
        """
        exp_dir = self.lab_artifacts_dir / experiment_id
        manifest_path = exp_dir / "experiment_manifest.json"
        if not manifest_path.exists():
            logger.error(f"Cannot execute job {experiment_id}: manifest not found.")
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            exp_manifest = json.load(f)

        start_time = time.time()
        exp_manifest["status"] = "running"
        self.write_experiment_manifest(exp_manifest)

        try:
            dataset_id = exp_manifest["dataset_id"]
            ds_manifest = exp_manifest["dataset_manifest"]
            model_ids = exp_manifest["model_ids"]
            seed = exp_manifest["seed"]

            csv_path = self.lab_data_dir / dataset_id / "dataset.csv"
            df = pd.read_csv(csv_path)

            feature_cols = ds_manifest["feature_columns"]
            label_col = ds_manifest["label_column"]
            weight_col = ds_manifest.get("weight_column")

            # Encode binary label to 0/1
            raw_labels = df[label_col].dropna().unique()
            pos_label = raw_labels[0]
            y = (df[label_col] == pos_label).astype(np.int32).to_numpy()
            X = df[feature_cols]

            weights = df[weight_col].to_numpy(dtype=np.float64) if weight_col else None

            # Split dataset 70% train, 15% val, 15% test
            X_train, X_rest, y_train, y_rest = train_test_split(
                X, y, test_size=0.30, random_state=seed, stratify=y
            )
            w_train, w_rest = None, None
            if weights is not None:
                w_train, w_rest = train_test_split(weights, test_size=0.30, random_state=seed, stratify=y)

            X_val, X_test, y_val, y_test = train_test_split(
                X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest
            )
            w_val, w_test = None, None
            if weights is not None and w_rest is not None:
                w_val, w_test = train_test_split(w_rest, test_size=0.50, random_state=seed, stratify=y_rest)

            per_model_results = {}

            for m_id in model_ids:
                # Check elapsed job timeout budget (300s)
                elapsed = time.time() - start_time
                if elapsed > settings.LAB_JOB_TIMEOUT_SECONDS:
                    exp_manifest["status"] = "failed"
                    exp_manifest["error_message"] = f"Job timeout exceeded ({elapsed:.1f}s > {settings.LAB_JOB_TIMEOUT_SECONDS}s cap)."
                    exp_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self.write_experiment_manifest(exp_manifest)
                    return

                logger.info(f"Training lab model '{m_id}' for experiment '{experiment_id}'")
                model_fit_start = time.time()

                # Build unfitted model pipeline
                pipeline = build_model(m_id, feature_set="all_physics")

                # Subsample cap for expensive models
                subsample_cap = self.MODEL_SUBSAMPLE_CAPS.get(m_id)
                X_tr_fit = X_train
                y_tr_fit = y_train
                actual_subsample_size = len(X_train)

                if subsample_cap and len(X_train) > subsample_cap:
                    actual_subsample_size = subsample_cap
                    _, X_tr_fit, _, y_tr_fit = train_test_split(
                        X_train, y_train, test_size=subsample_cap, random_state=seed, stratify=y_train
                    )

                # Fit on train split
                pipeline.fit(X_tr_fit, y_tr_fit)
                fit_duration = time.time() - model_fit_start

                # Predict probabilities on validation & test splits
                val_start = time.time()
                if hasattr(pipeline, "predict_proba"):
                    probs_val = pipeline.predict_proba(X_val)[:, 1]
                    probs_test = pipeline.predict_proba(X_test)[:, 1]
                else:
                    probs_val = pipeline.predict(X_val).astype(float)
                    probs_test = pipeline.predict(X_test).astype(float)

                inference_latency_ms = ((time.time() - val_start) / (len(X_val) + len(X_test))) * 1000.0

                # Threshold selection on validation split
                if weights is not None and w_val is not None:
                    dummy_weights_val = w_val
                    is_weighted_run = True
                else:
                    dummy_weights_val = np.ones(len(X_val), dtype=np.float64)
                    is_weighted_run = False

                opt_thresh, val_ams, _, _ = evaluate_threshold_scan(
                    y_val, probs_val, dummy_weights_val, num_thresholds=50
                )

                # Final evaluation on TEST split at optimal threshold
                selected_test = probs_test >= opt_thresh
                tp = int(((y_test == 1) & selected_test).sum())
                fp = int(((y_test == 0) & selected_test).sum())
                fn = int(((y_test == 1) & ~selected_test).sum())
                tn = int(((y_test == 0) & ~selected_test).sum())

                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                # Compute ROC-AUC on test split
                from sklearn.metrics import roc_auc_score
                try:
                    roc_auc = float(roc_auc_score(y_test, probs_test))
                except Exception:
                    roc_auc = 0.5

                # Compute AMS on test split if weighted
                if is_weighted_run and w_test is not None:
                    test_s = float(w_test[(y_test == 1) & selected_test].sum())
                    test_b = float(w_test[(y_test == 0) & selected_test].sum())
                    from ml.evaluation.metrics import compute_ams
                    test_ams = float(compute_ams(test_s, test_b))
                else:
                    test_s = None
                    test_b = None
                    test_ams = None

                result_contract = EvaluationResult(
                    model_id=m_id,
                    feature_set="custom_lab",
                    mode="lab_experiment",
                    roc_auc_mean=roc_auc,
                    f1_mean=f1,
                    optimal_threshold=float(opt_thresh),
                    ams_score=test_ams if test_ams is not None else 0.0,
                    training_duration_seconds=fit_duration,
                    inference_latency_ms=inference_latency_ms,
                    validation_rows=len(X_test),
                    precision_selected=prec,
                    recall_selected=rec,
                    confusion_matrix_selected={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    weighted_signal_yield_s=test_s if test_s is not None else 0.0,
                    weighted_background_yield_b=test_b if test_b is not None else 0.0,
                )

                # Persist model artifact strictly to models/lab_artifacts/{experiment_id}/{model_id}/
                model_artifact_dir = exp_dir / m_id
                model_artifact_dir.mkdir(parents=True, exist_ok=True)

                import joblib
                joblib.dump(pipeline, model_artifact_dir / "model.joblib")

                metrics_data = {
                    "test_metrics": result_contract.to_dict(),
                    "validation_optimal_threshold": float(opt_thresh),
                    "validation_ams": float(val_ams) if is_weighted_run else None,
                    "is_weighted": is_weighted_run,
                    "subsample_size": actual_subsample_size,
                }
                with open(model_artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, indent=2)

                per_model_results[m_id] = metrics_data

            exp_manifest["status"] = "completed"
            exp_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            exp_manifest["per_model_results"] = per_model_results
            self.write_experiment_manifest(exp_manifest)
            logger.info(f"Successfully completed experiment job '{experiment_id}'")

        except Exception as e:
            logger.exception(f"Experiment job '{experiment_id}' failed: {e}")
            exp_manifest["status"] = "failed"
            exp_manifest["error_message"] = str(e)
            exp_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.write_experiment_manifest(exp_manifest)


lab_job_runner = LabJobRunner()
