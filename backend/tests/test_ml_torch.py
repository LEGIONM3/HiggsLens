import numpy as np
import pytest
from ml.models.mlp_torch import TORCH_AVAILABLE, TorchMLPClassifier

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def test_torch_mlp_construction_and_cpu_fallback():
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment.")

    # CPU mode construction and fit on small synthetic fixture
    clf = TorchMLPClassifier(max_epochs=2, batch_size=16, device="cpu", seed=42)
    X = np.random.randn(30, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=30).astype(np.int64)

    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (30, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE or (torch is not None and not torch.cuda.is_available()), reason="CUDA GPU not available")
def test_torch_mlp_cuda_execution():
    clf = TorchMLPClassifier(max_epochs=2, batch_size=16, device="cuda", seed=42)
    X = np.random.randn(30, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=30).astype(np.int64)

    clf.fit(X, y)
    assert "cuda" in clf.actual_device_.lower()
    probs = clf.predict_proba(X)
    assert probs.shape == (30, 2)


def test_model_registry_torch_guard_without_crash():
    from backend.app.services.model_registry import ModelRegistryService
    registry = ModelRegistryService()
    # Scanning should never raise an exception even if torch model artifacts exist
    ids = registry.list_model_ids()
    assert isinstance(ids, list)
