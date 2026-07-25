"""
Standard Evaluation Metric Contract for HiggsLens ML Model Arena.
Shaped exactly like the backend's metrics.json artifact contract.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class EvaluationResult:
    """
    Standard metric contract emitted by model evaluation runs.
    Pointers and keys match models/artifacts/{model_id}/metrics.json verbatim.
    """

    model_id: str
    feature_set: str = "all_physics"
    mode: str = "fast"
    seeds_evaluated: List[int] = field(default_factory=lambda: [42])
    roc_auc_mean: float = 0.0
    roc_auc_std: float = 0.0
    pr_auc_mean: float = 0.0
    pr_auc_std: float = 0.0
    log_loss_mean: float = 0.0
    log_loss_std: float = 0.0
    balanced_accuracy_mean: float = 0.0
    balanced_accuracy_std: float = 0.0
    f1_mean: float = 0.0
    f1_std: float = 0.0
    brier_score_mean: float = 0.0
    brier_score_std: float = 0.0
    optimal_threshold: float = 0.6862
    ams_score: float = 0.0
    ams_default_threshold_score: float = 0.0
    training_duration_seconds: float = 0.0
    inference_latency_ms: float = 0.0
    stability_status: str = "not_assessed"
    calibration_status: str = "not_calibrated"
    validation_rows: int = 100000
    precision_05: float = 0.0
    recall_05: float = 0.0
    precision_selected: float = 0.0
    recall_selected: float = 0.0
    confusion_matrix_05: Dict[str, int] = field(default_factory=lambda: {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    confusion_matrix_selected: Dict[str, int] = field(default_factory=lambda: {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    weighted_signal_yield_s: float = 0.0
    weighted_background_yield_b: float = 0.0
    ams_br: float = 10.0
    calibration_method: str = "none"
    expected_calibration_error: float = 0.0
    reliability_bins: List[Dict[str, Any]] = field(default_factory=list)
    threshold_curve: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
