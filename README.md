# HiggsLens

**HiggsLens** is an educational and machine-learning platform for statistical Higgs-event classification on CERN/ATLAS open data (`CERN Open Data Record 328`). It compares signal and background topologies, evaluates classifier discrimination across pre-trained model artifacts, and provides an engineering control surface for reproducible physics ML inference.

> [!NOTE]
> **Scientific Disclaimer**: This application performs statistical Higgs-event classification on simulated collision events from the ATLAS Higgs Boson Machine Learning Challenge 2014 (`CERN Open Data Record 328`, DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`). It evaluates probability calibration and Approximate Median Significance (AMS). It does not discover the Higgs boson or claim new physical discovery from real collider measurements.

---

## Architecture Overview

HiggsLens features a decoupled vertical architecture separating web serving from offline ML design:

```
higgslens/
├── frontend/             # Vite 5 + React 18 + TypeScript + Tailwind CSS control surface
├── backend/              # Python 3.12 + FastAPI (inference-only) + Pydantic v2 + scikit-learn
│   ├── app/
│   │   ├── main.py       # create_app() factory, router mounting, CORS for Vite dev server
│   │   ├── core/         # config.py (pydantic-settings), logging.py
│   │   ├── schemas/      # dataset.py, models.py, predict.py, metrics.py
│   │   ├── services/     # dataset_service.py, model_registry.py, prediction_service.py, metrics_service.py
│   │   └── api/v1/       # health.py, dataset.py, models.py, predict.py, metrics.py
│   └── tests/            # Pytest test suite (45 unit/integration tests)
├── ml/                   # Offline ML Model Arena & Dataset Prep Foundations
│   ├── data/             # DatasetPrepPipeline, feature set specs, luminosity weight renormalization
│   ├── models/           # Declarative ModelSpec registry (11 candidate specs), build_model factory
│   └── evaluation/       # EvaluationResult contract & compute_ams / threshold scan utilities
├── scripts/              # Data downloader, dataset validator, & artifact migration script
├── configs/              # YAML configuration schemas
├── models/artifacts/     # Versioned pre-trained model artifacts (model.joblib, metrics.json, manifest.json)
└── artifacts/            # Source metrics logs and evaluation reports
```

---

## Model Arena Candidate Specifications (`ml/models/`)

The Model Arena defines declarative specifications (`ModelSpec`) and an unfitted pipeline factory (`build_model`) for 11 candidates:

| Candidate ID | Family | Display Name | Optional Dependencies | Experimental |
| :--- | :--- | :--- | :--- | :--- |
| `dummy_prior` | Baseline | Dummy Prior Baseline | None | No |
| `logistic_regression` | Linear | Logistic Regression | None | No |
| `random_forest` | Tree Ensemble | Random Forest | None | No |
| `histogram_gradient_boosting` | Boosting | Hist. Gradient Boosting | None | No |
| `mlp` | Neural Network | Multi-Layer Perceptron | None | No |
| `xgboost` | Boosting | XGBoost Classifier | `xgboost` | No |
| `lightgbm` | Boosting | LightGBM Classifier | `lightgbm` | No |
| `svm_rbf` | Support Vector | SVM (RBF Kernel) | None | No |
| `calibrated_ensemble` | Ensemble | Calibrated Voting Ensemble | None | No |
| `quantum_kernel_svm` | Quantum ML | Quantum Kernel SVM | `qiskit_machine_learning` | Yes (Stub) |
| `variational_quantum_classifier` | Quantum ML | Variational Quantum Classifier | `pennylane` | Yes (Stub) |

> [!NOTE]
> **QML Track Disclaimer**: The QML specifications are experimental interface stubs. They represent statistical classifiers benchmarked against classical baselines — never quantum simulation or quantum-randomness prediction.

---

## Backend API Endpoints

| Method | Endpoint | Description | Error Contracts |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Status, app version, count of available pre-trained model artifacts | — |
| `GET` | `/api/v1/dataset/summary` | Dataset facts: 818,238 events, 30 features, source "CERN/ATLAS open data record 328", DOI `10.7483/OPENDATA.ATLAS.ZBP2.M5T8` | — |
| `GET` | `/api/v1/models` | List registered pre-trained models with headline metrics from `metrics.json` | — |
| `GET` | `/api/v1/models/{model_id}/metrics` | Full stored metric set (ROC-AUC, AMS, precision, recall, F1, confusion matrix) | `404 Not Found` (unknown model_id) |
| `POST` | `/api/v1/predict` | Validated feature vector inference → probability, predicted label at threshold (default `0.6862`), manifest reference | `422 Unprocessable Entity` (invalid/missing features), `404 Not Found`, `503 Service Unavailable` (missing weights) |
| `GET` | `/api/v1/metrics/{model_id}/thresholds` | Stored AMS/threshold curve data points for frontend charting | `404 Not Found` |

---

## Candidate Model Performance Comparison (Validation Partition `b`, 100,000 rows)

Metrics below are persisted in `metrics.json` for each model artifact and served directly via `/api/v1/models`:

| Candidate Model (`model_id`) | ROC-AUC | PR-AUC | Log Loss | Balanced Acc | F1 Score | Brier Score | Opt. Thresh | Signal Yield `s` | Background Yield `b` | AMS ($b_r=10$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Prior** (`dummy_prior`) | 0.5000 | 0.3403 | 0.6412 | 0.5000 | 0.5077 | 0.2245 | 0.0100 | 84.25 | 50463.91 | 0.3749 |
| **Logistic Regression** (`logistic_regression`) | 0.8103 | 0.6692 | 0.5006 | 0.7265 | 0.6405 | 0.1651 | 0.4216 | 86.85 | 1000.00 | 0.6974 |
| **Random Forest** (`random_forest`) **[Recommended]** | **0.8851** | **0.8131** | **0.4051** | 0.7045 | 0.5866 | **0.1273** | **0.6862** | 33.41 | 989.07 | **1.0511** |
| **Hist. Gradient Boosting** (`histogram_gradient_boosting`) | 0.8828 | 0.8121 | 0.4091 | 0.7150 | 0.6069 | 0.1278 | 0.8136 | 33.76 | 1163.00 | 0.9810 |
| **PyTorch MLP Plugin** (`mlp`) | 0.8821 | 0.8043 | 0.4069 | 0.6795 | 0.5367 | 0.1290 | 0.8136 | 27.45 | 750.74 | 0.9894 |

---

## Quick Start & Execution

### 1. Run Artifact Migration
Package experiment metrics and pre-trained candidate model weights into `models/artifacts/`:
```bash
python scripts/migrate_artifacts.py
```

### 2. Start FastAPI Backend Server
```bash
uv run uvicorn backend.app.main:app --reload --port 8000
```
API Documentation is interactively available at `http://localhost:8000/docs`.

### 3. Run Quality Gates & Test Suite
```bash
# Run test suite (45 unit/integration tests)
uv run --python 3.12 pytest -v backend/tests

# Run Ruff linter
uv run --python 3.12 ruff check backend/app backend/tests scripts ml

# Run Mypy static type checker
uv run --python 3.12 mypy backend/app backend/tests scripts ml
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
