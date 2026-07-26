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

## 🏆 Model Arena Benchmark & Reports

- 📊 **R004 Initial Benchmark Report**: [reports/arena_benchmark_2026-07-25.md](file:///D:/HiggsLens/reports/arena_benchmark_2026-07-25.md)
- 🚀 **R005 GPU Uplift & Verification Report**: [reports/r005_uplift_2026-07-26.md](file:///D:/HiggsLens/reports/r005_uplift_2026-07-26.md)

### 📌 RF Parity Resolution & Historical Evidence (Dated Addendum — 2026-07-26)

> **RF Parity Note**: Historical baseline tables previously reported `random_forest` Test ROC-AUC of `0.8851`. Stored artifacts in commit `007f35b:models/artifacts/random_forest/metrics.json` confirm this `0.8851` result was produced by a **fast 100k subsample run** (`mode: "fast"`, `training_duration_seconds: 0.67s`, ~70k train / 30k val events). When trained on the full **250k train / 100k val / 450k test** Kaggle split (R004/R005), `random_forest` achieves **0.9061 Test ROC-AUC**, matching theoretical expectation.

### Latest Benchmark Leaderboard (R005 — 2026-07-26)

> **Dataset**: CERN/ATLAS open data record 328 · 818,238 events · SHA-256: `54242acf28a78ce3…`  
> **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120) · CUDA 13.2 · Driver 595.79  
> **Methodology**: Train (250k) → threshold on Val (100k) → final metrics on Test (450k)

| Model | Device | Val ROC-AUC | Test ROC-AUC | Test AMS | Status / Notes |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **XGBoost (CUDA)** | `cuda:0` 🚀 | 0.9111 | **0.9123** | 3.5754 | **Promoted (R005)** — 200 trees, depth 10, lr 0.03 |
| **LightGBM** | `cpu` | 0.9105 | **0.9114** | 3.5562 | **Promoted (R005)** — 200 trees, 63 leaves (PyPI CPU wheel) |
| **Hist. Gradient Boosting** | `cpu` | 0.9070 | 0.9085 | 3.5263 | Certified R004 artifact |
| **PyTorch MLP (sm_120 CUDA)** | `cuda:0` 🚀 | 0.9002 | 0.9014 | 3.2790 | 18-config sweep on GPU (R004 CPU baseline 0.9064 kept) |
| **Random Forest** | `cpu` | 0.9029 | 0.9061 | 3.4567 | Full Kaggle split (250k fit) |
| **Multi-Layer Perceptron (sklearn)** | `cpu` | 0.9009 | 0.9027 | 3.2119 | Certified R004 artifact |
| **Support Vector Machine (RBF)** | `cpu` | 0.8854 | **0.8868** | 2.9756 | **Promoted (R005)** — 50k subsample fit (551.4s) |
| **Calibrated Voting Ensemble** | `cpu` | 0.8943 | 0.8964 | 3.3693 | Certified R004 artifact |
| **Logistic Regression** | `cpu` | 0.8128 | 0.8147 | 2.0625 | Certified R004 artifact |
| **Dummy Prior Baseline** | `cpu` | 0.5000 | 0.5000 | 1.0791 | Certified R004 baseline |
| **Quantum Kernel SVM** | `cpu` (sim) | 0.4981 | 0.5012 | 1.0500 | Experimental Qiskit stub (100 subsample) |
| **Variational Quantum Classifier** | `cpu` (sim) | 0.5024 | 0.4978 | 1.0200 | Experimental PennyLane stub (100 subsample) |

> 🚀 **XGBoost CUDA**: Confirmed `device='cuda'`, fits in <2s on 250k events.  
> 🚀 **PyTorch CUDA**: Confirmed `cuda:0` (sm_120 RTX 5070 Ti) via `torch 2.13.0+cu132`.  
> ℹ️ **PennyLane Simulator**: `lightning.qubit` CPU simulator active (`lightning.gpu` requires Linux/WSL2).

