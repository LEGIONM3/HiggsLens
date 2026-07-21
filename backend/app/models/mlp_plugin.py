from typing import Any, List, Optional

import numpy as np
import pandas as pd
from backend.app.models.base import ModelCandidate
from backend.app.schemas.models import ModelInfo


class MLPCandidate(ModelCandidate):
    def __init__(self):
        super().__init__(
            model_id="mlp",
            display_name="Multi-Layer Perceptron (Small MLP)",
            required=False,
            preprocessing_strategy="impute_median_scale",
            supports_missing=False
        )
        self._status: Optional[str] = None
        self._backend: str = "sklearn"

    def check_dependency(self) -> str:
        if self._status is not None:
            return self._status
        try:
            import torch
            import torch.nn as nn
            self._backend = "torch"
            self._status = "available"
        except ImportError:
            try:
                from sklearn.neural_network import MLPClassifier
                self._backend = "sklearn"
                self._status = "available"
            except ImportError:
                self._status = "unavailable"
        except Exception:
            self._status = "incompatible"
        return self._status

    def get_info(self) -> ModelInfo:
        status = self.check_dependency()
        return ModelInfo(
            id=self.model_id,
            display_name=f"{self.display_name} ({self._backend.upper()})",
            status=status,
            required=self.required,
            supports_missing=self.supports_missing,
            preprocessing_pipeline=self.preprocessing_strategy,
            hyperparameters_schema={
                "hidden_layer_sizes": {"type": "list", "default": [64, 32], "description": "Hidden layer neurons"},
                "max_iter": {"type": "int", "default": 200, "description": "Max training epochs/iterations"},
                "alpha": {"type": "float", "default": 0.0001, "description": "L2 regularization"}
            }
        )

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray, feature_set: str = "all_physics", **hyperparameters: Any) -> "MLPCandidate":
        status = self.check_dependency()
        if status != "available":
            raise RuntimeError("MLP dependencies (PyTorch or scikit-learn) are not available.")

        hidden_sizes = hyperparameters.get("hidden_layer_sizes", [64, 32])
        if isinstance(hidden_sizes, str):
            hidden_sizes = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
        max_iter = int(hyperparameters.get("max_iter", 200))
        alpha = float(hyperparameters.get("alpha", 0.0001))
        seed = int(hyperparameters.get("random_state", 42))

        X_train = self.pipeline.fit_transform(df_train, feature_set)

        if self._backend == "torch":
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            torch.manual_seed(seed)
            in_features = X_train.shape[1]
            layers: List[nn.Module] = []
            prev_size = in_features
            for size in hidden_sizes:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                prev_size = size
            layers.append(nn.Linear(prev_size, 2))

            self.model = nn.Sequential(*layers)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=alpha)

            dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=256, shuffle=True)

            self.model.train()
            for epoch in range(min(max_iter, 50)):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    out = self.model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
        else:
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_sizes),
                max_iter=max_iter,
                alpha=alpha,
                random_state=seed,
                early_stopping=True
            )
            self.model.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X = self.pipeline.transform(df)
        if self._backend == "torch":
            import torch
            self.model.eval()
            with torch.no_grad():
                logits = self.model(torch.tensor(X, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1).numpy()
            return probs
        else:
            return self.model.predict_proba(X)
