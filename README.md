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

## Dataset & Citation

We use the official ATLAS Higgs Boson Machine Learning Challenge 2014 dataset from CERN Open Data:

- **Source URL**: [https://opendata.cern.ch/record/328](https://opendata.cern.ch/record/328)
- **DOI**: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`
- **Citation**: ATLAS collaboration (2014). *Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014*. CERN Open Data Portal. DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`
- **Characteristics**: 818,238 simulated events (`186.5 MiB` compressed `.csv.gz`), containing 30 primary detector (`PRI_*`) and 13 derived physics (`DER_*`) variables. Signal simulations fix the Higgs boson mass at $125\text{ GeV}$.

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
| **Test Suite** | `make test` | Runs backend `pytest` fixtures/evaluations and frontend `vitest` suite |
| **Lint / Types** | `make lint` | Runs `ruff` checks, `mypy` strict checking, and TypeScript validation |
| **Data Download**| `make download-data` | Downloads `atlas-higgs-challenge-2014-v2.csv.gz` from official CERN record |
| **Fast Training**| `make train-fast` | Trains baseline models on a fast-mode bounded stratified sample |

---

## License

This project is licensed under the MIT License. See [LICENSE](file:///d:/HiggsLens/LICENSE) for details.
