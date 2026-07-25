"""
Declarative ModelSpec Interface for HiggsLens ML Model Arena.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class DependencyMissingError(Exception):
    """Raised when attempting to build a model whose optional requirement is not installed."""

    def __init__(self, model_id: str, required_pkg: str):
        self.model_id = model_id
        self.required_pkg = required_pkg
        super().__init__(
            f"Model '{model_id}' requires package '{required_pkg}', which is not installed. "
            f"Install optional extras via 'pip install .[ml]'."
        )


@dataclass
class ModelSpec:
    """
    Declarative specification for a candidate model in the HiggsLens ML Model Arena.
    """

    model_id: str
    family: str  # "baseline", "linear", "tree_ensemble", "boosting", "neural_network", "support_vector_machine", "ensemble", "quantum_ml"
    display_name: str
    hyperparameter_search_space: Dict[str, Any]
    metadata: Dict[str, Any]
    builder: Callable[[str], Any]
    requires: List[str] = field(default_factory=list)
    experimental: bool = False
    description: str = ""

    def build(self, feature_set: str = "all_physics") -> Any:
        """Constructs an unfitted sklearn-compatible Pipeline for the requested feature set."""
        return self.builder(feature_set)
