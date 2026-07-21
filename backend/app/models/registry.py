from typing import Any, Dict

from backend.app.models.base import ModelCandidate
from backend.app.models.dummy import DummyPriorCandidate
from backend.app.models.hgb import HistGradientBoostingCandidate
from backend.app.models.logistic import LogisticRegressionCandidate
from backend.app.models.mlp_plugin import MLPCandidate
from backend.app.models.rf import RandomForestCandidate
from backend.app.models.xgb_plugin import XGBoostCandidate
from backend.app.schemas.models import ModelInfo


class ModelRegistry:
    """Registry for all HiggsLens model candidates."""
    def __init__(self):
        self._classes: Dict[str, Any] = {
            "dummy_prior": DummyPriorCandidate,
            "logistic_regression": LogisticRegressionCandidate,
            "random_forest": RandomForestCandidate,
            "histogram_gradient_boosting": HistGradientBoostingCandidate,
            "xgboost": XGBoostCandidate,
            "mlp": MLPCandidate,
        }
        self._instances: Dict[str, ModelCandidate] = {}

    def get_candidate(self, model_id: str) -> ModelCandidate:
        if model_id not in self._classes:
            raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(self._classes.keys())}")
        if model_id not in self._instances:
            self._instances[model_id] = self._classes[model_id]()
        return self._instances[model_id]

    def create_fresh_candidate(self, model_id: str) -> ModelCandidate:
        if model_id not in self._classes:
            raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(self._classes.keys())}")
        return self._classes[model_id]()

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for m_id in self._classes.keys():
            cand = self.get_candidate(m_id)
            info = cand.get_info()
            result[m_id] = info.model_dump()
        return result

    def get_model_info(self, model_id: str) -> ModelInfo:
        cand = self.get_candidate(model_id)
        return cand.get_info()


model_registry = ModelRegistry()
