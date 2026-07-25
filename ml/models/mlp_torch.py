"""
PyTorch Multi-Layer Perceptron Classifier for HiggsLens ML Model Arena.
Supports CUDA acceleration on NVIDIA GPUs, with fallback handling for compute capability mismatches.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("higgslens.ml.mlp_torch")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None     # type: ignore
    optim = None  # type: ignore


class _PyTorchMLPModule(nn.Module if TORCH_AVAILABLE else object):  # type: ignore
    def __init__(self, input_dim: int, hidden_layer_sizes: tuple = (64, 32), num_classes: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    scikit-learn compatible PyTorch Multi-Layer Perceptron Classifier.
    Executes training and prediction on CUDA GPU (NVIDIA RTX 5070 Ti) when available.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        max_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 0.001,
        seed: int = 42,
        device: str = "cuda"
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device
        self.classes_: Optional[np.ndarray] = None
        self.module_: Optional[Any] = None
        self.input_dim_: Optional[int] = None
        self.actual_device_: str = "cpu"

    def _verify_torch_installed(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install optional dependencies via 'pip install .[ml]'.")

    def fit(self, X: Any, y: Any) -> "TorchMLPClassifier":
        self._verify_torch_installed()

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_arr = y.to_numpy(dtype=np.int64)
        else:
            y_arr = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y_arr)
        self.input_dim_ = X_arr.shape[1]

        torch.manual_seed(self.seed)

        target_device = torch.device("cpu")
        self.actual_device_ = "cpu"

        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TorchMLPClassifier: device='cuda' requested but torch.cuda.is_available() "
                    "is False. Install a CUDA-enabled torch wheel."
                )
            try:
                candidate_device = torch.device("cuda:0")
                # Validate GPU actually executes (catches sm_120 / kernel image errors)
                t_test = torch.randn(5, 5, device=candidate_device)
                _ = t_test @ t_test
                target_device = candidate_device
                cc = torch.cuda.get_device_capability(0)
                name = torch.cuda.get_device_name(0)
                self.actual_device_ = f"cuda:0 ({name}, sm_{cc[0]}{cc[1]})"
                logger.info(f"TorchMLPClassifier: training on {self.actual_device_}")
            except Exception as e:
                raise RuntimeError(
                    f"TorchMLPClassifier: CUDA execution failed on this GPU — {e}. "
                    "For RTX 5070 Ti (sm_120/Blackwell), install: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu132"
                ) from e

        self.module_ = _PyTorchMLPModule(self.input_dim_, self.hidden_layer_sizes, len(self.classes_)).to(target_device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.module_.parameters(), lr=self.lr)

        dataset_size = len(X_arr)
        indices = np.arange(dataset_size)

        self.module_.train()
        for epoch in range(self.max_epochs):
            np.random.seed(self.seed + epoch)
            np.random.shuffle(indices)

            for i in range(0, dataset_size, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                bx = torch.tensor(X_arr[batch_idx], dtype=torch.float32, device=target_device)
                by = torch.tensor(y_arr[batch_idx], dtype=torch.long, device=target_device)

                optimizer.zero_grad()
                outputs = self.module_(bx)
                loss = criterion(outputs, by)
                loss.backward()
                optimizer.step()

        logger.info(f"Completed TorchMLPClassifier training on {self.actual_device_} ({self.max_epochs} epochs)")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self._verify_torch_installed()
        if self.module_ is None:
            raise RuntimeError("TorchMLPClassifier is not fitted yet.")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        target_device = next(self.module_.parameters()).device
        self.module_.eval()

        with torch.no_grad():
            bx = torch.tensor(X_arr, dtype=torch.float32, device=target_device)
            logits = self.module_(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def build_mlp_torch_pipeline(feature_set: str = "all_physics") -> Pipeline:
    """Builds an unfitted StandardScaler + TorchMLPClassifier pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", TorchMLPClassifier(hidden_layer_sizes=(64, 32), max_epochs=50, batch_size=256, seed=42, device="cuda"))
    ])
