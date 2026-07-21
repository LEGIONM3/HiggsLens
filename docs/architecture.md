# HiggsLens Architecture Documentation

## 1. Monorepo Structural Diagram

```
higgslens/
├── frontend/             # Next.js 15, React 19, TypeScript, Tailwind CSS, TanStack Query
├── backend/              # Python 3.12, FastAPI, Pydantic v2, scikit-learn, Polars/Pandas
├── scripts/              # Data downloading, validation, and CLI model training
├── configs/              # YAML configurations (dataset, features, models, experiments)
├── data/                 # Local data storage across raw, interim, and processed stages
├── artifacts/            # Model registry artifacts, metrics JSON, SQLite db, model cards
└── docs/                 # Scientific documentation and specifications
```

## 2. Data Flow & Preprocessing Pipeline
1. **Streaming Download**: `scripts/download_dataset.py` streams `atlas-higgs-challenge-2014-v2.csv.gz` from CERN Open Data into `data/raw/atlas-higgs-challenge-2014-v2.csv.gz.part` before atomically renaming to `.csv.gz`.
2. **Schema & Partitioning**: `backend.app.data.pipeline` parses columns, enforces `KaggleSet` partitions (`t` $\to$ train, `b` $\to$ validation, `v` $\to$ test, `u` $\to$ holdout), and validates `-999.0` sentinel distributions against jet multiplicity (`PRI_jet_num`).
3. **Feature Grouping & Preprocessing**: Features are grouped (`primary_only`, `derived_only`, `all_physics`). Imputers (`median`) and scalers (`StandardScaler`) are fit strictly on training partitions to prevent data leakage. Missingness indicators are appended when configured.

## 3. Training & Model Arena Flow
- **Model Registry**: Dynamic registry (`backend.app.models.registry`) manages candidates (`dummy_prior`, `logistic_regression`, `random_forest`, `histogram_gradient_boosting`, and optional `xgboost`/`mlp`).
- **Modes**:
  - `fast_mode`: Trains candidate models on a bounded stratified sample (`5,000` rows default) using a single seed (`42`) for quick end-to-end engineering verification.
  - `research_mode`: Trains across repeated deterministic seeds (`[42, 123, 2026]`) on the full `t` partition, reporting metric mean/variance and computing optimal `AMS` decision thresholds on `b` (validation).
- **Experiment Tracking**: Lightweight local tracking logs run config, timestamps, metrics, duration, and generated Markdown model cards (`artifacts/models/*.md`) to an index (`artifacts/experiments.sqlite` or `.jsonl`).

## 4. Prediction & Event-Visualization Contract
When the frontend requests `/api/v1/predict`, the response includes:
- Probabilities and threshold classification.
- An **Event Visualization Payload** containing strictly reconstructed physical objects supported by primary features (`tau`, `lepton`, `MET`, `jets`). No unmeasured particle trajectories or fabricated physics tracks are included.
