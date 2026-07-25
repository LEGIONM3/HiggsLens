"""
Declarative Model Spec Registry for HiggsLens ML Model Arena.
Contains all 12 registered candidate specifications (5 baselines, 5 classical additions, 2 experimental QML stubs).
"""

import importlib
import logging
from typing import Any, Dict
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.models.spec import DependencyMissingError, ModelSpec

logger = logging.getLogger("higgslens.ml.registry")


# --- Builder Callbacks ---

def _build_dummy_prior(feature_set: str) -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def _build_logistic_regression(feature_set: str) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
    ])


def _build_random_forest(feature_set: str) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)


def _build_histogram_gradient_boosting(feature_set: str) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=30, random_state=42)


def _build_mlp(feature_set: str) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, early_stopping=True, random_state=42))
    ])


def _build_mlp_torch(feature_set: str) -> Any:
    try:
        from ml.models.mlp_torch import build_mlp_torch_pipeline
        return build_mlp_torch_pipeline(feature_set)
    except ImportError:
        raise DependencyMissingError("mlp_torch", "torch")


def _build_xgboost(feature_set: str) -> Any:
    try:
        xgb = importlib.import_module("xgboost")
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            device="cuda",
            tree_method="hist"
        )
    except ImportError:
        raise DependencyMissingError("xgboost", "xgboost")


def _build_lightgbm(feature_set: str) -> Any:
    try:
        lgb = importlib.import_module("lightgbm")
        import numpy as _np
        # Probe GPU support with a tiny synthetic fit — the GPU fatal error
        # only surfaces at fit time, not at LGBMClassifier() construction.
        _X_probe = _np.random.RandomState(0).randn(20, 4)
        _y_probe = _np.array([0] * 10 + [1] * 10)
        try:
            _probe = lgb.LGBMClassifier(
                n_estimators=2, max_depth=2, learning_rate=0.1,
                random_state=42, device="gpu", verbose=-1
            )
            _probe.fit(_X_probe, _y_probe)
            logger.info("LightGBM GPU probe succeeded — using device='gpu'.")
            return lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, device="gpu", verbose=-1
            )
        except Exception as _e:
            logger.warning(
                f"LightGBM GPU probe failed ({_e}); "
                "falling back to CPU. Recompile with -DUSE_GPU=1 for GPU support."
            )
            return lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, device="cpu", verbose=-1, n_jobs=-1
            )
    except ImportError:
        raise DependencyMissingError("lightgbm", "lightgbm")


def _build_svm_rbf(feature_set: str) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, kernel="rbf", C=1.0, random_state=42))
    ])


def _build_calibrated_ensemble(feature_set: str) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=50, random_state=42)),
            ("lr", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=42, n_jobs=-1))]))
        ],
        voting="soft",
        n_jobs=-1
    )


def _build_quantum_kernel_svm(feature_set: str) -> Any:
    try:
        qiskit_machine = importlib.import_module("qiskit_machine_learning")
        return qiskit_machine.algorithms.QSVC()
    except ImportError:
        raise DependencyMissingError("quantum_kernel_svm", "qiskit_machine_learning")


def _build_variational_quantum_classifier(feature_set: str) -> Any:
    try:
        pennylane = importlib.import_module("pennylane")
        return pennylane
    except ImportError:
        raise DependencyMissingError("variational_quantum_classifier", "pennylane")


# --- Declarative Model Specifications Registry ---

MODEL_SPECS: Dict[str, ModelSpec] = {
    "dummy_prior": ModelSpec(
        model_id="dummy_prior",
        family="baseline",
        display_name="Dummy Prior Baseline",
        hyperparameter_search_space={"strategy": ["prior"]},
        metadata={"interpretability": "high", "expected_cost": "trivial", "supports_missing_natively": True, "device": "cpu"},
        builder=_build_dummy_prior,
        requires=[],
        experimental=False,
        description="Naive baseline predicting target class prior probability."
    ),
    "logistic_regression": ModelSpec(
        model_id="logistic_regression",
        family="linear",
        display_name="Logistic Regression",
        hyperparameter_search_space={"model__C": [0.01, 0.1, 1.0, 10.0], "model__solver": ["lbfgs"]},
        metadata={"interpretability": "high", "expected_cost": "low", "supports_missing_natively": False, "device": "cpu"},
        builder=_build_logistic_regression,
        requires=[],
        experimental=False,
        description="L2-regularized linear decision boundary with standardized features."
    ),
    "random_forest": ModelSpec(
        model_id="random_forest",
        family="tree_ensemble",
        display_name="Random Forest",
        hyperparameter_search_space={"n_estimators": [50, 100, 200], "max_depth": [10, 15, 20]},
        metadata={"interpretability": "medium", "expected_cost": "medium", "supports_missing_natively": True, "device": "cpu"},
        builder=_build_random_forest,
        requires=[],
        experimental=False,
        description="Ensemble of decision trees with feature subsampling (Recommended baseline)."
    ),
    "histogram_gradient_boosting": ModelSpec(
        model_id="histogram_gradient_boosting",
        family="boosting",
        display_name="Hist. Gradient Boosting",
        hyperparameter_search_space={"max_iter": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.2]},
        metadata={"interpretability": "medium", "expected_cost": "medium", "supports_missing_natively": True, "device": "cpu"},
        builder=_build_histogram_gradient_boosting,
        requires=[],
        experimental=False,
        description="Fast histogram-based gradient boosting decision trees with native missing value support."
    ),
    "mlp": ModelSpec(
        model_id="mlp",
        family="neural_network",
        display_name="Multi-Layer Perceptron (MLP)",
        hyperparameter_search_space={"model__hidden_layer_sizes": [(64, 32), (128, 64)], "model__alpha": [0.0001, 0.001]},
        metadata={"interpretability": "low", "expected_cost": "high", "supports_missing_natively": False, "device": "cpu"},
        builder=_build_mlp,
        requires=[],
        experimental=False,
        description="Multi-layer perceptron neural network (scikit-learn CPU implementation)."
    ),
    "mlp_torch": ModelSpec(
        model_id="mlp_torch",
        family="neural_network",
        display_name="PyTorch MLP (CUDA)",
        hyperparameter_search_space={"model__hidden_layer_sizes": [(64, 32)], "model__lr": [0.001]},
        metadata={"interpretability": "low", "expected_cost": "medium", "supports_missing_natively": False, "device": "cuda:0 (RTX 5070 Ti)"},
        builder=_build_mlp_torch,
        requires=["torch"],
        experimental=False,
        description="PyTorch MLP classifier with CUDA acceleration on NVIDIA GeForce RTX 5070 Ti."
    ),
    "xgboost": ModelSpec(
        model_id="xgboost",
        family="boosting",
        display_name="XGBoost Classifier (CUDA)",
        hyperparameter_search_space={"n_estimators": [50, 100, 200], "max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1]},
        metadata={"interpretability": "medium", "expected_cost": "medium", "supports_missing_natively": True, "device": "cuda:0 (RTX 5070 Ti)"},
        builder=_build_xgboost,
        requires=["xgboost"],
        experimental=False,
        description="Scalable gradient boosting with CUDA tree_method=hist acceleration."
    ),
    "lightgbm": ModelSpec(
        model_id="lightgbm",
        family="boosting",
        display_name="LightGBM Classifier",
        hyperparameter_search_space={"n_estimators": [50, 100, 200], "num_leaves": [31, 63], "learning_rate": [0.01, 0.1]},
        metadata={"interpretability": "medium", "expected_cost": "low", "supports_missing_natively": True, "device": "gpu/cpu"},
        builder=_build_lightgbm,
        requires=["lightgbm"],
        experimental=False,
        description="Fast histogram-based gradient boosting framework."
    ),
    "svm_rbf": ModelSpec(
        model_id="svm_rbf",
        family="support_vector_machine",
        display_name="Support Vector Machine (RBF)",
        hyperparameter_search_space={"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
        metadata={"interpretability": "low", "expected_cost": "high", "supports_missing_natively": False, "device": "cpu"},
        builder=_build_svm_rbf,
        requires=[],
        experimental=False,
        description="Subsample-aware support vector classification with Radial Basis Function kernel."
    ),
    "calibrated_ensemble": ModelSpec(
        model_id="calibrated_ensemble",
        family="ensemble",
        display_name="Calibrated Voting Ensemble",
        hyperparameter_search_space={"voting": ["soft"]},
        metadata={"interpretability": "low", "expected_cost": "high", "supports_missing_natively": False, "device": "cpu"},
        builder=_build_calibrated_ensemble,
        requires=[],
        experimental=False,
        description="Soft voting ensemble aggregating probability predictions from Random Forest, HistGB, and Logistic Regression."
    ),
    "quantum_kernel_svm": ModelSpec(
        model_id="quantum_kernel_svm",
        family="quantum_ml",
        display_name="Quantum Kernel Support Vector Machine",
        hyperparameter_search_space={"C": [0.1, 1.0, 10.0]},
        metadata={"interpretability": "low", "expected_cost": "experimental", "supports_missing_natively": False, "device": "cpu/gpu"},
        builder=_build_quantum_kernel_svm,
        requires=["qiskit_machine_learning"],
        experimental=True,
        description="Statistical Higgs-event classifier benchmarked against classical baselines — never quantum simulation or quantum-randomness prediction."
    ),
    "variational_quantum_classifier": ModelSpec(
        model_id="variational_quantum_classifier",
        family="quantum_ml",
        display_name="Variational Quantum Classifier",
        hyperparameter_search_space={"num_layers": [2, 4], "step_size": [0.01, 0.05]},
        metadata={"interpretability": "low", "expected_cost": "experimental", "supports_missing_natively": False, "device": "cpu/gpu"},
        builder=_build_variational_quantum_classifier,
        requires=["pennylane"],
        experimental=True,
        description="Statistical Higgs-event classifier benchmarked against classical baselines — never quantum simulation or quantum-randomness prediction."
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    """Returns ModelSpec for requested model_id."""
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model_id '{model_id}'. Registered: {list(MODEL_SPECS.keys())}")
    return MODEL_SPECS[model_id]


def list_model_specs() -> Dict[str, ModelSpec]:
    """Returns all registered model specifications."""
    return MODEL_SPECS
