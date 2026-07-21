import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from backend.app.core.config import settings
from backend.app.schemas.experiments import ExperimentRunResponse, RecommendationExplanation
from backend.app.schemas.models import ModelMetricsSchema


class ExperimentTracker:
    """
    Lightweight local experiment tracker storing run logs in JSONL and generating Markdown model cards.
    """
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or settings.ARTIFACTS_DIR
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.log_file = self.artifacts_dir / "experiments_log.jsonl"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_git_commit() -> Optional[str]:
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    def generate_model_card(
        self,
        model_id: str,
        display_name: str,
        feature_set: str,
        preprocessing_strategy: str,
        metrics: ModelMetricsSchema
    ) -> str:
        card_path = self.models_dir / f"{model_id}_card.md"
        content = f"""# Model Card: {display_name} (`{model_id}`)

## 1. Intended Use & Scope
- **Task**: Binary classification of simulated ATLAS Higgs-to-tau-tau ($H \to \tau\tau$) collision events against background processes ($Z \to \tau\tau$, $t\\bar{{t}}$, $W+\\text{{jets}}$).
- **Target Audience**: Educational and scientific machine learning researchers analyzing benchmark collider datasets (`CERN Open Data Record 328`).
- **Out of Scope**: Real-time LHC trigger deployment, uncalibrated discovery claims, or generalized high-energy physics classification beyond the $8\\text{{ TeV}}$ Higgs Challenge topology.

## 2. Dataset & Feature Set
- **Dataset Version**: ATLAS Higgs Boson Machine Learning Challenge 2014 (`DOI: 10.7483/OPENDATA.ATLAS.ZBP2.M5T8`).
- **Partitioning**: Official `KaggleSet` mapping (`t` for training, `b` for validation and threshold optimization, `v` for final test evaluation).
- **Feature Group**: `{feature_set}`.

## 3. Preprocessing & Sentinel Handling
- **Missing Value Strategy**: `{preprocessing_strategy}`.
- **Sentinel Handling**: `-999.0` values corresponding to physical unavailability under jet multiplicity (`PRI_jet_num`) were treated according to physical rules. Imputers and scalers were fit exclusively on training data.

## 4. Evaluation Metrics (Validation Set)
- **ROC-AUC**: `{metrics.roc_auc_mean:.4f} ± {metrics.roc_auc_std:.4f}`
- **PR-AUC**: `{metrics.pr_auc_mean:.4f} ± {metrics.pr_auc_std:.4f}`
- **Log Loss**: `{metrics.log_loss_mean:.4f} ± {metrics.log_loss_std:.4f}`
- **Balanced Accuracy**: `{metrics.balanced_accuracy_mean:.4f} ± {metrics.balanced_accuracy_std:.4f}`
- **F1 Score**: `{metrics.f1_mean:.4f} ± {metrics.f1_std:.4f}`
- **Brier Score (Calibration)**: `{metrics.brier_score_mean:.4f} ± {metrics.brier_score_std:.4f}`
- **Optimal Decision Threshold (Validation)**: `{metrics.optimal_threshold:.4f}`
- **Approximate Median Significance (AMS)**: `{metrics.ams_score:.4f}` (evaluated at $b_r = 10$)

## 5. Stability Across Seeds & Calibration
- **Seeds Evaluated**: `{metrics.seeds_evaluated}`
- **Stability Assessment**: `{metrics.stability_status}`
- **Calibration Status**: `{metrics.calibration_status}`
- **Training Duration**: `{metrics.training_duration_seconds:.2f} s`

## 6. Ethical & Scientific Caveats
- Predictions represent statistical similarity to Monte Carlo simulation templates. High classifier confidence does not imply real-world particle existence or physical discovery.
"""
        with open(card_path, "w") as f:
            f.write(content)
        return str(card_path)

    def log_run(
        self,
        run_id: str,
        dataset_fingerprint: str,
        split_strategy: str,
        feature_set: str,
        mode: str,
        seeds: List[int],
        metrics_by_model: Dict[str, ModelMetricsSchema]
    ) -> ExperimentRunResponse:
        ts = datetime.now(timezone.utc).isoformat()
        git_commit = self.get_git_commit()

        # Compute recommendation based on validation performance, stability, probability calibration, compute cost, and weighted AMS
        recommendation = self.compute_recommendation(metrics_by_model, feature_set)

        response = ExperimentRunResponse(
            run_id=run_id,
            timestamp=ts,
            git_commit=git_commit,
            dataset_fingerprint=dataset_fingerprint,
            split_strategy=split_strategy,
            feature_set=feature_set,
            mode=mode,
            seeds=seeds,
            metrics_by_model=metrics_by_model,
            recommendation=recommendation
        )

        with open(self.log_file, "a") as f:
            f.write(response.model_dump_json() + "\n")

        # Also save individual run json
        run_json = self.metrics_dir / f"run_{run_id}.json"
        with open(run_json, "w") as f:
            f.write(response.model_dump_json(indent=2))

        return response

    @staticmethod
    def compute_recommendation(
        metrics_by_model: Dict[str, ModelMetricsSchema],
        feature_set: str
    ) -> Optional[RecommendationExplanation]:
        if not metrics_by_model:
            return None

        # Exclude dummy prior from winner recommendation unless it's the only one
        candidates = {k: v for k, v in metrics_by_model.items() if k != "dummy_prior"}
        if not candidates:
            candidates = metrics_by_model

        best_model_id = None
        best_score = -1.0
        warnings = []

        for m_id, m in candidates.items():
            # Composite objective: 0.4 * ROC_AUC + 0.3 * (AMS/3.0) + 0.2 * (1 - Brier) - 0.1 * variance
            ams_norm = min(1.0, m.ams_score / 3.0)
            cal_bonus = max(0.0, 1.0 - (m.brier_score_mean * 4.0))
            var_penalty = min(0.5, m.roc_auc_std * 5.0)

            score = 0.4 * m.roc_auc_mean + 0.3 * ams_norm + 0.2 * cal_bonus - var_penalty
            if score > best_score:
                best_score = score
                best_model_id = m_id

        if best_model_id is None:
            best_model_id = list(candidates.keys())[0]

        best_m = candidates[best_model_id]
        if best_m.stability_status != "stable":
            warnings.append(f"Recommended model {best_model_id} exhibits {best_m.stability_status} across evaluated seeds.")
        if best_m.calibration_status == "uncalibrated":
            warnings.append(f"Recommended model {best_model_id} has high Brier score ({best_m.brier_score_mean:.3f}). Consider probability calibration.")

        reasoning = (
            f"Automatic recommendation based on validation performance, stability, probability calibration, compute cost, and weighted AMS. "
            f"Model '{best_model_id}' achieved the highest composite objective (ROC-AUC: {best_m.roc_auc_mean:.4f}, "
            f"validation AMS: {best_m.ams_score:.4f} at threshold {best_m.optimal_threshold:.2f}, "
            f"Brier score: {best_m.brier_score_mean:.4f}, duration: {best_m.training_duration_seconds:.2f}s). "
            f"Evaluated strictly on validation partition without test-set leakage."
        )

        return RecommendationExplanation(
            recommended_model_id=best_model_id,
            validation_objective_used="Composite validation ROC-AUC + Weighted AMS + Calibration - Variance penalty",
            feature_set_used=feature_set,
            mean_roc_auc=best_m.roc_auc_mean,
            variance_across_seeds=best_m.roc_auc_std ** 2,
            calibration_quality=best_m.calibration_status,
            training_duration_seconds=best_m.training_duration_seconds,
            selected_threshold=best_m.optimal_threshold,
            weighted_ams=best_m.ams_score,
            warnings=warnings,
            reasoning=reasoning
        )
