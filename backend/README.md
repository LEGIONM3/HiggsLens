# HiggsLens Backend

Python 3.12 FastAPI backend for the HiggsLens scientific machine-learning platform.

## Features
- **Data Pipeline**: Downloads `atlas-higgs-challenge-2014-v2.csv.gz` from CERN Open Data (`record/328`), enforces partition split rules (`t`, `b`, `v`, `u`), and handles jet multiplicity sentinels (`-999.0`).
- **Model Arena & Registry**: Manages baseline classifiers (`DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`) and checks availability of optional plugins (`XGBoost`, `PyTorch` MLP).
- **Physics Evaluation**: Evaluates ROC-AUC, PR-AUC, calibration, and Approximate Median Significance (`AMS`) across decision thresholds.
- **REST API**: Provides endpoints for dataset status, validation, model registry inspection, training triggers, real-time job polling, and prediction with event visualization payloads (`Hadronic tau`, `Lepton`, `MET`, `Jets`).

## Running Locally
```bash
uv venv --python 3.12 .venv
uv pip install -e backend/
uv run --python 3.12 uvicorn backend.app.main:app --reload --port 8000
```
