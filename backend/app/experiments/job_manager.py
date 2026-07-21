import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from backend.app.data.pipeline import DatasetPipeline
from backend.app.data.validator import DatasetValidator
from backend.app.evaluation.evaluator import ModelEvaluator
from backend.app.experiments.tracker import ExperimentTracker
from backend.app.models.registry import model_registry
from backend.app.schemas.experiments import JobStatusResponse, TrainingRequest
from backend.app.schemas.models import ModelMetricsSchema


class BackgroundJobManager:
    """
    Lightweight in-process background-job manager for running model training experiments asynchronously.
    Does not require external Redis or Celery.
    """
    def __init__(self):
        self._jobs: Dict[str, JobStatusResponse] = {}
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.tracker = ExperimentTracker()

    def create_job(self, request: TrainingRequest) -> JobStatusResponse:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Determine target models
        if request.models:
            target_models = request.models
        else:
            # Train all available models
            target_models = [
                m_id for m_id, info in model_registry.list_models().items()
                if info["status"] == "available"
            ]

        job = JobStatusResponse(
            job_id=job_id,
            state="queued",
            current_model=None,
            completed_models=0,
            total_models=len(target_models),
            progress_message="Job queued for execution.",
            started_timestamp=now,
            updated_timestamp=now,
            error_details=None,
            run_id=None
        )
        with self._lock:
            self._jobs[job_id] = job

        # Launch background worker thread
        thread = threading.Thread(target=self._run_training_job, args=(job_id, request, target_models), daemon=True)
        thread.start()
        return job

    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_result(self, job_id: str) -> Optional[Any]:
        with self._lock:
            return self._results.get(job_id)

    def list_jobs(self) -> List[JobStatusResponse]:
        with self._lock:
            return list(self._jobs.values())

    def _update_job(self, job_id: str, state: str, message: str, current_model: Optional[str] = None, completed_models: Optional[int] = None, error: Optional[str] = None, run_id: Optional[str] = None):
        with self._lock:
            if job_id in self._jobs:
                j = self._jobs[job_id]
                j.state = state
                j.progress_message = message
                j.updated_timestamp = datetime.now(timezone.utc).isoformat()
                if current_model is not None:
                    j.current_model = current_model
                if completed_models is not None:
                    j.completed_models = completed_models
                if error is not None:
                    j.error_details = error
                if run_id is not None:
                    j.run_id = run_id

    def _run_training_job(self, job_id: str, request: TrainingRequest, target_models: List[str]):
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        try:
            self._update_job(job_id, "validating_data", "Loading dataset and verifying schema partitions...")
            pipeline = DatasetPipeline()
            validator = DatasetValidator()

            # Determine seeds
            seeds = request.seeds or ([42] if request.mode == "fast" else [42, 123, 2026])

            # If fast mode, take bounded stratified sample (e.g. 5000 rows)
            sample_size = 5000 if request.mode == "fast" else None
            partitions = pipeline.get_partition_splits(
                strategy="official_partition",
                seed=seeds[0],
                sample_size=sample_size
            )

            df_train = partitions["train"]
            df_val = partitions["validation"]

            # Fingerprint
            fingerprint = validator.compute_fingerprint(df_train)

            self._update_job(job_id, "preprocessing", f"Preprocessing feature set '{request.feature_set}' across {len(df_train):,} training events...")

            y_train = DatasetPipeline.encode_labels(df_train)

            metrics_by_model: Dict[str, ModelMetricsSchema] = {}

            for idx, m_id in enumerate(target_models):
                self._update_job(
                    job_id,
                    "training",
                    f"Training model {idx+1}/{len(target_models)}: {m_id} across seeds {seeds}...",
                    current_model=m_id,
                    completed_models=idx
                )

                roc_aucs, pr_aucs, log_losses, bal_accs, f1s, briers, threshs, amss, ams_05s = [], [], [], [], [], [], [], [], []
                t0 = time.time()

                last_candidate = None
                for seed in seeds:
                    cand = model_registry.create_fresh_candidate(m_id)
                    cand.fit(df_train, y_train, feature_set=request.feature_set, random_state=seed)
                    last_candidate = cand

                    self._update_job(job_id, "evaluating", f"Evaluating {m_id} (seed {seed}) on validation partition...")
                    eval_metrics = ModelEvaluator.evaluate_model_on_val(cand, df_val, feature_set=request.feature_set)

                    roc_aucs.append(eval_metrics["roc_auc"])
                    pr_aucs.append(eval_metrics["pr_auc"])
                    log_losses.append(eval_metrics["log_loss"])
                    bal_accs.append(eval_metrics["balanced_accuracy"])
                    f1s.append(eval_metrics["f1"])
                    briers.append(eval_metrics["brier_score"])
                    threshs.append(eval_metrics["optimal_threshold"])
                    amss.append(eval_metrics["ams_score"])
                    ams_05s.append(eval_metrics["ams_05_score"])

                duration = time.time() - t0

                # Check variance stability across seeds
                roc_std = float(np.std(roc_aucs))
                if len(seeds) <= 1:
                    stability = "not_assessed"
                else:
                    stability = "stable" if roc_std < 0.05 else "high_variance"

                # Check calibration
                brier_mean = float(np.mean(briers))
                cal_status = "not_calibrated"
                cal_method = "none"

                last_eval = eval_metrics if isinstance(eval_metrics, dict) else {}

                m_schema = ModelMetricsSchema(
                    model_id=m_id,
                    feature_set=request.feature_set,
                    mode=request.mode,
                    seeds_evaluated=seeds,
                    roc_auc_mean=float(np.mean(roc_aucs)),
                    roc_auc_std=roc_std,
                    pr_auc_mean=float(np.mean(pr_aucs)),
                    pr_auc_std=float(np.std(pr_aucs)),
                    log_loss_mean=float(np.mean(log_losses)),
                    log_loss_std=float(np.std(log_losses)),
                    balanced_accuracy_mean=float(np.mean(bal_accs)),
                    balanced_accuracy_std=float(np.std(bal_accs)),
                    f1_mean=float(np.mean(f1s)),
                    f1_std=float(np.std(f1s)),
                    brier_score_mean=brier_mean,
                    brier_score_std=float(np.std(briers)),
                    optimal_threshold=float(np.mean(threshs)),
                    ams_score=float(np.mean(amss)),
                    ams_default_threshold_score=float(np.mean(ams_05s)),
                    training_duration_seconds=duration,
                    stability_status=stability,
                    calibration_status=cal_status,
                    validation_rows=last_eval.get("validation_rows", 0),
                    precision_05=last_eval.get("precision_05", 0.0),
                    recall_05=last_eval.get("recall_05", 0.0),
                    precision_selected=last_eval.get("precision_selected", 0.0),
                    recall_selected=last_eval.get("recall_selected", 0.0),
                    confusion_matrix_05=last_eval.get("confusion_matrix_05", {}),
                    confusion_matrix_selected=last_eval.get("confusion_matrix_selected", {}),
                    weighted_signal_yield_s=last_eval.get("weighted_signal_yield_s", 0.0),
                    weighted_background_yield_b=last_eval.get("weighted_background_yield_b", 0.0),
                    ams_br=last_eval.get("ams_br", 10.0),
                    calibration_method=cal_method,
                    expected_calibration_error=last_eval.get("expected_calibration_error", 0.0),
                    reliability_bins=last_eval.get("reliability_bins", [])
                )

                # Generate model card & save checkpoint
                self._update_job(job_id, "saving_artifacts", f"Generating model card and artifact for {m_id}...")
                info = model_registry.get_candidate(m_id).get_info()
                card_path = self.tracker.generate_model_card(
                    m_id, info.display_name, request.feature_set, cand.preprocessing_strategy, m_schema
                )
                m_schema.model_card_path = card_path
                metrics_by_model[m_id] = m_schema

                # Also save last fitted model instance in registry so live prediction endpoint can use it immediately!
                if last_candidate is not None:
                    model_registry._instances[m_id] = last_candidate

            self._update_job(job_id, "saving_artifacts", "Finalizing run index and recommendation log...", completed_models=len(target_models))

            run_response = self.tracker.log_run(
                run_id=run_id,
                dataset_fingerprint=fingerprint,
                split_strategy="official_partition",
                feature_set=request.feature_set,
                mode=request.mode,
                seeds=seeds,
                metrics_by_model=metrics_by_model
            )

            with self._lock:
                self._results[job_id] = run_response

            self._update_job(
                job_id,
                "completed",
                f"Training completed successfully across {len(target_models)} models! Recommended: {run_response.recommendation.recommended_model_id if run_response.recommendation else 'N/A'}",
                current_model=None,
                completed_models=len(target_models),
                run_id=run_id
            )

        except Exception as e:
            import traceback
            err_msg = f"Job failed: {str(e)}\n{traceback.format_exc()}"
            self._update_job(job_id, "failed", f"Failed: {str(e)}", error=err_msg)


job_manager = BackgroundJobManager()
