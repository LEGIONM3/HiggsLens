# HiggsLens

**HiggsLens** is an educational and machine-learning platform for statistical Higgs-event classification on CERN/ATLAS open data (`CERN Open Data Record 328`). It compares signal and background topologies, evaluates classifier discrimination across pre-trained certified model artifacts, and provides an engineering control surface for reproducible physics ML inference and sandboxed custom experimentation.

> [!NOTE]
> **Scientific Disclaimer**: This application performs statistical Higgs-event classification on simulated collision events from the ATLAS Higgs Boson Machine Learning Challenge 2014 (`CERN Open Data Record 328`, DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`). It evaluates probability calibration and Approximate Median Significance (AMS). Pre-trained certified weights are benchmarked on CERN/ATLAS open data. HiggsLens does not discover the Higgs boson or claim new physical discovery from real collider measurements.

---

## Architecture Overview

HiggsLens features a decoupled vertical architecture separating certified pre-trained inference serving from the sandboxed Lab experimentation zone:

```
higgslens/
├── frontend/             # Vite 5 + React 18 + TypeScript + Tailwind CSS control surface
├── backend/              # Python 3.12 + FastAPI + Pydantic v2 + scikit-learn
│   ├── app/
│   │   ├── main.py       # create_app() factory, router mounting, CORS for Vite dev server
│   │   ├── core/         # config.py (pydantic-settings), logging.py
│   │   ├── schemas/      # dataset.py, models.py, predict.py, metrics.py, lab.py
│   │   ├── services/     # dataset_service.py, model_registry.py, prediction_service.py, metrics_service.py, lab/
│   │   └── api/v1/       # health.py, dataset.py, models.py, predict.py, metrics.py, lab/
│   └── tests/            # Pytest test suite (50 unit/integration tests)
├── ml/                   # Offline ML Model Arena & Dataset Prep Foundations
│   ├── data/             # DatasetPrepPipeline, feature set specs, luminosity weight renormalization
│   ├── models/           # Declarative ModelSpec registry (11 candidate specs), build_model factory
│   └── evaluation/       # EvaluationResult contract & compute_ams / threshold scan utilities
├── data/
│   ├── raw/              # CERN ATLAS open data raw CSV
│   └── lab/              # Sandboxed user-uploaded custom datasets (UUID dataset_id)
├── models/
│   ├── artifacts/        # Certified pre-trained model artifacts (READ-ONLY, isolated)
│   └── lab_artifacts/    # Custom experiment artifacts & disk-persisted manifests
├── scripts/              # Data downloader, dataset validator, & artifact migration script
└── configs/              # YAML configuration schemas
```

---

## HiggsLens Lab Sandboxed Zone API Endpoints (`/api/v1/lab`)

HiggsLens Lab allows users to upload custom datasets and train candidate models in an isolated sandbox. Certified pre-trained model weights in `models/artifacts/` remain strictly isolated and untouched.

| Method | Endpoint | Description | Resource Caps / Error Contracts |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/lab/datasets` | Upload custom CSV dataset with column mapping parameters | Max 200 MB file size cap, Max 500k row cap, UUID `dataset_id`, Binary label check → `422` |
| `GET` | `/api/v1/lab/datasets` | List uploaded custom lab datasets and manifests | — |
| `POST` | `/api/v1/lab/experiments` | Submit custom model training experiment job | Max 5 models per experiment, Max 1 concurrent job → `409 Conflict`, 300s timeout → `422`/`503` |
| `GET` | `/api/v1/lab/experiments` | List custom lab experiment training jobs | Reads disk-persisted `experiment_manifest.json` |
| `GET` | `/api/v1/lab/experiments/{id}` | Retrieve lab experiment status and detailed leaderboard | Reports final `EvaluationResult` metrics on `test` split at validation optimal threshold |

---

## Model Arena Candidate Specifications (`ml/models/`)

| Candidate ID | Family | Display Name | Optional Requirements | Build Status |
| :--- | :--- | :--- | :--- | :--- |
| `dummy_prior` | Baseline | Dummy Prior Baseline | None | **OK** |
| `logistic_regression` | Linear | Logistic Regression | None | **OK** |
| `random_forest` | Tree Ensemble | Random Forest | None | **OK** |
| `histogram_gradient_boosting` | Boosting | Hist. Gradient Boosting | None | **OK** |
| `mlp` | Neural Network | Multi-Layer Perceptron | None | **OK** |
| `xgboost` | Boosting | XGBoost Classifier | `xgboost` | **Skipped cleanly if uninstalled** |
| `lightgbm` | Boosting | LightGBM Classifier | `lightgbm` | **Skipped cleanly if uninstalled** |
| `svm_rbf` | Support Vector | SVM (RBF Kernel) | None | **OK (Subsampled to 10k rows)** |
| `calibrated_ensemble` | Ensemble | Calibrated Voting Ensemble | None | **OK** |
| `quantum_kernel_svm` | Quantum ML | Quantum Kernel SVM | `qiskit_machine_learning` | **Experimental Stub** |
| `variational_quantum_classifier` | Quantum ML | Variational Quantum Classifier | `pennylane` | **Experimental Stub** |

---

## Quick Start & Execution

### 1. Start FastAPI Backend Server
```bash
uv run uvicorn backend.app.main:app --reload --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 2. Start Frontend Dev Server
```bash
cd frontend && npm run dev
```

### 3. Run Quality Gates & Test Suite
```bash
# Run backend test suite (50 unit/integration tests)
uv run --python 3.12 pytest -v backend/tests

# Run Ruff linter
uv run --python 3.12 ruff check backend/app backend/tests scripts ml

# Run Mypy static type checker
uv run --python 3.12 mypy backend/app backend/tests scripts ml

# Run Frontend typecheck and unit tests
npm --prefix frontend run typecheck
npm --prefix frontend test
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
