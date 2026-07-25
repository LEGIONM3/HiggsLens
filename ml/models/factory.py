"""
Factory function for constructing unfitted sklearn-compatible candidate pipelines.
"""

import importlib.util
from typing import Any

from ml.models.registry import get_model_spec
from ml.models.spec import DependencyMissingError, ModelSpec


def build_model(model_id: str, feature_set: str = "all_physics") -> Any:
    """
    Factory function returning an unfitted sklearn-compatible Pipeline or Estimator for requested candidate model.
    Does NOT invoke .fit() or perform training orchestration.
    Raises DependencyMissingError if optional package dependencies are missing.
    """
    spec: ModelSpec = get_model_spec(model_id)

    # Verify required optional packages
    for req_pkg in spec.requires:
        spec_found = importlib.util.find_spec(req_pkg)
        if spec_found is None:
            raise DependencyMissingError(model_id, req_pkg)

    return spec.build(feature_set)
