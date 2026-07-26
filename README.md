# HiggsLens

**HiggsLens** is an educational and machine-learning research platform for statistical Higgs-event classification on CERN/ATLAS open data (`CERN Open Data Record 328`). It compares signal and background topologies, evaluates classifier discrimination across pre-trained certified model artifacts, and provides an interactive engineering control surface featuring a 3D Event Display, LHC Accelerator Journey, Official Model Leaderboard, Curated Event Gallery, and sandboxed custom experimentation zone.

[![CI Status](https://github.com/LEGIONM3/HiggsLens/actions/workflows/ci.yml/badge.svg)](https://github.com/LEGIONM3/HiggsLens/actions/workflows/ci.yml)

> [!NOTE]
> **Scientific Disclaimer**: This application performs statistical Higgs-event classification on simulated collision events from the ATLAS Higgs Boson Machine Learning Challenge 2014 (`CERN Open Data Record 328`, DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`). It evaluates probability calibration and Approximate Median Significance (AMS). Pre-trained certified weights are benchmarked on CERN/ATLAS open data. HiggsLens does not discover the Higgs boson or claim new physical discovery from real collider measurements.

---

## Architecture Overview

HiggsLens features a decoupled vertical architecture separating certified pre-trained inference serving from the sandboxed Lab experimentation zone:

```
higgslens/
├── frontend/             # Vite 5 + React 18 + TypeScript + Three.js / @react-three/fiber + Tailwind CSS
│   ├── src/
│   │   ├── components/   # display/ (3D Display), journey/ (LHC Ring), leaderboard/, gallery/, education/, common/
│   │   ├── context/      # EducationContext.tsx (3-level physics guide state persistence)
│   │   ├── lib/          # kinematics.ts (pure 3D physics library), thresholdExplorer.ts, sentinel.ts
│   │   └── services/     # api.ts (typed FastAPI client)
├── backend/              # Python 3.12 + FastAPI + Pydantic v2 + scikit-learn + XGBoost TreeSHAP
│   ├── app/
│   │   ├── main.py       # create_app() factory, router mounting, CORS for Vite dev server
│   │   ├── schemas/      # dataset.py, models.py, predict.py, metrics.py, derive.py, explain.py, gallery.py, lab.py
│   │   ├── services/     # event_sampling.py, prediction_service.py, derivation.py, explanation.py, gallery.py, lab/
│   │   └── api/v1/       # health.py, dataset.py, models.py, predict.py, metrics.py, events.py, explain.py, lab/
│   └── tests/            # Pytest test suite (87 automated unit/integration tests)
├── ml/                   # Offline ML Model Arena & Dataset Prep Foundations
│   ├── data/             # DatasetPrepPipeline, feature set specs, luminosity weight renormalization
│   ├── models/           # Declarative ModelSpec registry (12 candidate specs), build_model factory
│   └── evaluation/       # EvaluationResult contract & compute_ams / threshold scan utilities
├── data/
│   ├── raw/              # CERN ATLAS open data raw CSV (atlas-higgs-challenge-2014-v2.csv.gz)
│   └── processed/v1/     # Processed splits (train.csv, validation.csv, test.csv, holdout.csv)
├── models/
│   └── artifacts/        # 12 Certified pre-trained model artifacts (READ-ONLY, frozen)
└── scripts/              # Data downloader, dataset validator, & update_model_manifests.py script
```

---

## Key Backend Serving Endpoints (`/api/v1`)

| Method | Endpoint | Description | Status / Caps |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status and dataset readiness | 200 OK |
| `GET` | `/api/v1/models` | List all 12 certified models with headline metrics & manifests | Dynamic from `metrics.json` |
| `POST` | `/api/v1/predict` | Score 30-feature vector using certified model | Certified path |
| `GET` | `/api/v1/events/sample` | Sample test-split ATLAS events deterministically | Max 50 cap, holdout isolated |
| `POST` | `/api/v1/events/derive` | Re-derive 13 calculated features from 17 primary inputs | Pure kinematics calculator |
| `POST` | `/api/v1/explain` | Native XGBoost TreeSHAP feature attributions | Additivity gate validated ($10^{-6}$) |
| `GET` | `/api/v1/events/gallery` | Curated gallery of signal, background, and interesting events | CI fixture sidecar fallback |
| `GET` | `/api/v1/events/{id}/permalink` | Cold-load event + explanation in single API call | Holdout isolated (404 on `u`) |

---

## Model Arena Candidate Specifications (`ml/models/`)

| Candidate ID | Family | Display Name | Optional Requirements | Status & Benchmark Performance |
| :--- | :--- | :--- | :--- | :--- |
| `xgboost` | Boosting | XGBoost Classifier | `xgboost` | **Certified Champion** — ROC-AUC 0.9123, AMS 3.5754 |
| `lightgbm` | Boosting | LightGBM Classifier | `lightgbm` | **Runner-Up** — ROC-AUC 0.9114, AMS 3.5562 |
| `histogram_gradient_boosting` | Boosting | Hist. Gradient Boosting | None | **OK** — ROC-AUC 0.9085, AMS 3.5263 |
| `random_forest` | Tree Ensemble | Random Forest | None | **OK** — ROC-AUC 0.9061, AMS 3.4567 |
| `mlp` | Neural Network | Multi-Layer Perceptron | None | **OK** — ROC-AUC 0.9027, AMS 3.2119 |
| `svm_rbf` | Support Vector | SVM (RBF Kernel) | None | **OK** — ROC-AUC 0.8868, AMS 2.9756 (CPU, 50k subsample) |
| `calibrated_ensemble` | Ensemble | Calibrated Voting Ensemble | None | **OK** — ROC-AUC 0.8964, AMS 3.3693 |
| `logistic_regression` | Linear | Logistic Regression | None | **OK** — ROC-AUC 0.8147, AMS 2.0625 |
| `dummy_prior` | Baseline | Dummy Prior Baseline | None | **OK** — ROC-AUC 0.5000, AMS 1.0791 |
| `quantum_kernel_svm` | Quantum ML | Quantum Kernel SVM | `qiskit` | **Research Stub** — ROC-AUC 0.5012, AMS 1.0500 |
| `variational_quantum_classifier` | Quantum ML | Variational Quantum Classifier | `pennylane` | **Research Stub** — ROC-AUC 0.4978, AMS 1.0200 |

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

### 3. Run Automated Test Suite (118 Tests)
```bash
# Run backend pytest suite (87 automated unit/integration tests)
uv run --python 3.12 pytest -v backend/tests

# Run Ruff linter
uv run --python 3.12 ruff check backend/app backend/tests scripts ml

# Run Mypy static type checker
uv run --python 3.12 mypy backend/app backend/tests scripts ml

# Run Frontend typecheck and Vitest suite (31 automated tests)
npm --prefix frontend run typecheck
npm --prefix frontend test
```

---

## 🏆 Official Frozen Model Leaderboard

> **Dataset Provenance**: ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8) — official ATLAS simulated events, classified by certified pre-trained models.

| Model | Device | Val ROC-AUC | Test ROC-AUC | Test AMS | Optimal Threshold |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **XGBoost (CUDA)** | `cuda:0` | 0.9111 | **0.9123** | **3.5754** | `0.8118` |
| **LightGBM** | `CPU` | 0.9105 | **0.9114** | 3.5562 | `0.8050` |
| **Hist. Gradient Boosting** | `CPU` | 0.9070 | 0.9085 | 3.5263 | `0.7920` |
| **Random Forest** | `CPU` | 0.9029 | 0.9061 | 3.4567 | `0.7850` |
| **Multi-Layer Perceptron (sklearn)** | `CPU` | 0.9009 | 0.9027 | 3.2119 | `0.7620` |
| **Calibrated Voting Ensemble** | `CPU` | 0.8943 | 0.8964 | 3.3693 | `0.7740` |
| **Support Vector Machine (RBF)** | `CPU (50k subsample)` | 0.8854 | 0.8868 | 2.9756 | `0.7410` |
| **Logistic Regression** | `CPU` | 0.8128 | 0.8147 | 2.0625 | `0.6520` |
| **Dummy Prior Baseline** | `CPU` | 0.5000 | 0.5000 | 1.0791 | `0.5000` |
| **Quantum Kernel SVM** | `CPU (sim)` | 0.4981 | 0.5012 | 1.0500 | `0.5000` |
| **Variational Quantum Classifier** | `CPU (sim)` | 0.5024 | 0.4978 | 1.0200 | `0.5000` |

> 📌 **RF 0.8851 Footnote**: Historical fast-mode benchmark runs scored `0.8851` on a 100k subsample. Full 250k training fit achieves `0.9061` Test ROC-AUC.  
> ⚛️ **QML Research Disclaimer**: Quantum ML models (`qml_vqc`, `qml_qaoa`) are experimental research benchmarks evaluated on a 100-event budget that scored at chance level on this tabular task — an honest negative result reported deliberately.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
