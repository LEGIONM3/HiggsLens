# HiggsLens

**HiggsLens** is a research-scale educational and machine-learning platform for exploring simulated ATLAS Higgs-to-tau-tau ($H \to \tau\tau$) collision events, comparing signal and background processes, training and evaluating multiple classifiers across repeated random seeds, and providing an engineering control surface for reproducible physics ML.

> [!NOTE]
> **Scientific Disclaimer**: This application analyzes simulated data from the ATLAS Higgs Boson Machine Learning Challenge 2014 (`CERN Open Data Record 328`). It establishes classifier discrimination (`signal-like` vs `background-like` topologies) and evaluates probability calibration and Approximate Median Significance (AMS). It does not discover the Higgs boson or claim new physical discovery from real collider measurements.

---

## Architecture Overview

HiggsLens follows a clean vertical monorepo design separating scientific computation from the browser interface:

```
higgslens/
├── frontend/             # Next.js 15 + React 19 + TypeScript + Tailwind CSS control surface
├── backend/              # Python 3.12 + FastAPI + Pydantic v2 + scikit-learn/Polars/Pandas
├── scripts/              # Data downloading, validation, and CLI model training
├── configs/              # YAML configuration schemas (dataset, features, models, experiments)
├── data/                 # Raw/interim/processed data stages (excluded from Git)
├── artifacts/            # Model registries, metrics, evaluation reports, and model cards
└── docs/                 # Scientific scope, dataset card, terminology, and architecture
```

---

## Dataset, Partitioning & Sentinel Handling

We use the official ATLAS Higgs Boson Machine Learning Challenge 2014 dataset from CERN Open Data:

- **Source URL**: [https://opendata.cern.ch/record/328](https://opendata.cern.ch/record/328)
- **DOI**: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`
- **Citation**: ATLAS collaboration (2014). *Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014*. CERN Open Data Portal. DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`
- **Characteristics**: 818,238 simulated events (`186.5 MiB` compressed `.csv.gz`), containing 30 primary detector (`PRI_*`) and 13 derived physics (`DER_*`) variables. Signal simulations fix the Higgs boson mass at $125\text{ GeV}$.
- **Partitioning Rule (`KaggleSet`)**:
  - `KaggleSet == 't'` (250,000 events): Exclusively used for training model parameters, imputers, scalers, and probability calibrators.
  - `KaggleSet == 'b'` (100,000 complete events): Validation partition used strictly for threshold optimization ($s, b$ yields and AMS score evaluation) and model comparison. Zero test-set leakage.
  - `KaggleSet == 'v'` (468,238 events): Holdout test partition reserved for final unblinded assessment.
- **Sentinel & Missing Values**: Missing detector measurements represented by `-999.0` (such as jet sub-variables when `PRI_jet_num == 0`) are handled deterministically (`impute_median` or `native` tree splits) fitted strictly on the training partition (`t`).

---

## Candidate Model Performance Comparison (Validation Partition `b`, 100,000 rows)

> [!IMPORTANT]
> **FAST-MODE SAMPLE RESULT — NOT A FULL-DATASET BENCHMARK**: The table below reports validation partition metrics evaluated after training on a fast-mode stratified training sample (`seeds=[42]`).

| Candidate Model (`model_id`) | ROC-AUC | PR-AUC | Log Loss | Balanced Acc | F1 Score | Brier Score | ECE | Opt. Thresh | Signal Yield `s` | Background Yield `b` | AMS ($b_r=10$) | Calibration Status | Stability Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Prior** (`dummy_prior`) | 0.5000 | 0.3427 | 0.6421 | 0.5000 | 0.0000 | 0.2253 | 0.0000 | 0.3427 | 691.99 | 1000.00 | 0.0000 | not_calibrated | not_assessed |
| **Logistic Regression** (`logistic_regression`) | 0.7303 | 0.5376 | 0.5693 | 0.6030 | 0.2255 | 0.1947 | 0.1654 | 0.6974 | 86.85 | 1000.00 | 0.2238 | not_calibrated | not_assessed |
| **Random Forest** (`random_forest`) **[Recommended]** | **0.8851** | **0.8131** | **0.4051** | 0.7045 | 0.5866 | **0.1273** | 0.0436 | 0.6862 | 33.41 | 989.07 | **1.0511** | not_calibrated | not_assessed |
| **Hist. Gradient Boosting** (`histogram_gradient_boosting`) | 0.8828 | 0.8121 | 0.4091 | **0.7150** | **0.6069** | 0.1278 | 0.0423 | 0.8136 | 33.76 | 1163.00 | 0.9810 | not_calibrated | not_assessed |
| **PyTorch MLP Plugin** (`mlp`) | 0.8821 | 0.8043 | 0.4069 | 0.6795 | 0.5367 | 0.1290 | **0.0186** | 0.8136 | 27.45 | 750.74 | 0.9894 | not_calibrated | not_assessed |

### Key Observations
- **AMS Threshold Optimization**: While threshold 0.5 yields standard accuracy, maximizing the Approximate Median Significance (AMS) shifts the optimal threshold higher ($0.6862$ for Random Forest and $0.8136$ for Hist. Gradient Boosting) to suppress background yield ($b$) and maximize collider discovery sensitivity.
- **Calibration & Stability**: Single-seed (`seeds=[42]`) fast-mode runs report `not_assessed` stability and `not_calibrated` status (`calibration_method="none"`). When post-training calibration (Platt scaling or isotonic regression) is applied across multi-seed runs, `calibration_status` transitions to `calibrated`.

---

## Quick Start & Setup

We manage the Python runtime deterministically with `uv` (pinned to Python 3.12) and Node dependencies via `npm`.

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/LEGIONM3/HiggsLens.git
cd HiggsLens

# Copy configuration template
cp .env.example .env

# Initialize environment and install dependencies
make setup
```

### 2. Download & Validate CERN Dataset
```bash
# Download official dataset directly from CERN Open Data (supports progress, caching, atomic rename)
make download-data

# Run comprehensive data validation and generate schema/sentinel check reports
make validate-data
```

### 3. Model Training Arena
```bash
# Train models in fast_mode (bounded stratified sample across candidate models)
make train-fast

# Or run full research_mode across repeated seeds [42, 123, 2026]
uv run --python 3.12 python scripts/train_models.py --mode research
```

### 4. Start Development Servers
```bash
# Start FastAPI backend (port 8000) and Next.js frontend (port 3000) concurrently
make dev
```
Visit `http://localhost:3000` to access the Overview, Dataset Inspector, Model Arena, and Prediction Sandbox.

---

## Verification & Commands

| Task | Command | Description |
| :--- | :--- | :--- |
| **Test Suite** | `make test` or `uv run --python 3.12 python -m pytest -v` | Runs all 24 unit/integration tests verifying API, physics AMS logic, bounds checks, and partition integrity |
| **Lint / Types** | `make lint` or `uv run --python 3.12 python -m ruff check .` | Runs `ruff` checks, `mypy` strict checking, and TypeScript validation |
| **Data Download**| `make download-data` | Downloads `atlas-higgs-challenge-2014-v2.csv.gz` from official CERN record |
| **Fast Training**| `make train-fast` | Trains baseline candidates on a fast-mode bounded stratified sample (`seeds=[42]`) |

---

## License

This project is licensed under the MIT License. See [LICENSE](file:///d:/HiggsLens/LICENSE) for details.
