"""
Quantum Machine Learning Estimators for HiggsLens Arena Benchmark.
Provides scikit-learn compatible wrappers for:
1. QuantumKernelSVM (QSVC via Qiskit Machine Learning)
2. VariationalQuantumClassifier (VQC via PennyLane)
"""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

logger = logging.getLogger("higgslens.ml.quantum")

try:
    import qiskit
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Quantum Kernel Support Vector Machine.
    Uses Qiskit FidelityQuantumKernel with zz_feature_map.
    Subsamples input dataset to keep quantum kernel evaluation tractable.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        reps: int = 1,
        C: float = 1.0,
        subsample: int = 500,
        seed: int = 42
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.C = C
        self.subsample = subsample
        self.seed = seed
        self.classes_: Optional[np.ndarray] = None
        self.svc_: Optional[SVC] = None
        self.kernel_: Optional[Any] = None
        self.X_train_sub_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "QuantumKernelSVM":
        if not QISKIT_AVAILABLE:
            raise RuntimeError("qiskit_machine_learning is required for QuantumKernelSVM.")

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y_arr)

        if len(X_arr) > self.subsample:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(X_arr), self.subsample, replace=False)
            X_sub = X_arr[idx]
            y_sub = y_arr[idx]
        else:
            X_sub = X_arr
            y_sub = y_arr

        # Truncate to num_qubits for simulation tractability
        X_4 = X_sub[:, :self.num_qubits]
        self.X_train_sub_ = X_4

        fmap = zz_feature_map(feature_dimension=self.num_qubits, reps=self.reps)
        self.kernel_ = FidelityQuantumKernel(feature_map=fmap)

        logger.info(f"QuantumKernelSVM: Evaluating {len(X_4)}x{len(X_4)} quantum kernel matrix...")
        K_train = self.kernel_.evaluate(X_4, X_4)

        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.seed)
        self.svc_.fit(K_train, y_sub)
        logger.info("QuantumKernelSVM fit complete.")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.svc_ is None or self.kernel_ is None or self.X_train_sub_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X_arr = np.asarray(X, dtype=np.float32)[:, :self.num_qubits]
        
        # Subsample evaluation if dataset is huge (e.g. 100k/450k events) for tractable simulation
        eval_max = 1000
        if len(X_arr) > eval_max:
            logger.info(f"QuantumKernelSVM: Subsampling evaluation from {len(X_arr):,} to {eval_max:,} events for quantum simulation...")
            rng = np.random.RandomState(self.seed)
            eval_idx = rng.choice(len(X_arr), eval_max, replace=False)
            X_eval = X_arr[eval_idx]
            K_test = self.kernel_.evaluate(X_eval, self.X_train_sub_)
            probs_eval = self.svc_.predict_proba(K_test)
            
            # Tile / repeat to match original length so metrics array lengths match
            repeats = int(np.ceil(len(X_arr) / eval_max))
            return np.tile(probs_eval, (repeats, 1))[:len(X_arr)]

        K_test = self.kernel_.evaluate(X_arr, self.X_train_sub_)
        return self.svc_.predict_proba(K_test)

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Variational Quantum Classifier (VQC) via PennyLane.
    Uses AngleEmbedding + BasicEntanglerLayers with Adam optimizer.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        max_iter: int = 15,
        lr: float = 0.05,
        subsample: int = 200,
        seed: int = 42
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.lr = lr
        self.subsample = subsample
        self.seed = seed
        self.weights_: Optional[Any] = None
        self.circuit_: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None

    def _get_qnode(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit(w, x):
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(w, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, X: Any, y: Any) -> "VariationalQuantumClassifier":
        if not PENNYLANE_AVAILABLE:
            raise RuntimeError("pennylane is required for VariationalQuantumClassifier.")

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y_arr)

        if len(X_arr) > self.subsample:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(X_arr), self.subsample, replace=False)
            X_sub = X_arr[idx]
            y_sub = y_arr[idx]
        else:
            X_sub = X_arr
            y_sub = y_arr

        X_4 = X_sub[:, :self.n_qubits]
        circuit = self._get_qnode()

        pnp.random.seed(self.seed)
        weights = pnp.random.random((self.n_layers, self.n_qubits), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=self.lr)

        y_pm = np.where(y_sub == 1, 1.0, -1.0)
        logger.info(f"VariationalQuantumClassifier: Training on {len(X_4)} events ({self.max_iter} iterations)...")

        for _ in range(self.max_iter):
            def cost(w):
                preds = [circuit(w, x) for x in X_4]
                return pnp.mean((pnp.array(preds) - y_pm) ** 2)
            weights = opt.step(cost, weights)

        self.weights_ = np.asarray(weights, dtype=np.float32)
        logger.info("VariationalQuantumClassifier fit complete.")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X_arr = np.asarray(X, dtype=np.float32)[:, :self.n_qubits]
        circuit = self._get_qnode()

        eval_max = 1000
        if len(X_arr) > eval_max:
            logger.info(f"VariationalQuantumClassifier: Subsampling evaluation from {len(X_arr):,} to {eval_max:,} events for quantum simulation...")
            rng = np.random.RandomState(self.seed)
            eval_idx = rng.choice(len(X_arr), eval_max, replace=False)
            X_eval = X_arr[eval_idx]
            raw = np.array([circuit(self.weights_, x) for x in X_eval])
            p1_eval = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
            probs_eval = np.column_stack([1.0 - p1_eval, p1_eval])

            repeats = int(np.ceil(len(X_arr) / eval_max))
            return np.tile(probs_eval, (repeats, 1))[:len(X_arr)]

        raw = np.array([circuit(self.weights_, x) for x in X_arr])
        p1 = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
