# HiggsLens Data Directory

This directory stores raw, interim, and processed data artifacts for HiggsLens.

> [!IMPORTANT]
> **Git Exclusion**: Raw and processed datasets are intentionally excluded from Git repository history via `.gitignore` to prevent committing massive multi-hundred megabyte data files (`atlas-higgs-challenge-2014-v2.csv.gz`).

---

## Official CERN Open Data Source

We use the canonical release from CERN Open Data:
- **Record**: [https://opendata.cern.ch/record/328](https://opendata.cern.ch/record/328)
- **DOI**: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`
- **Citation**: ATLAS collaboration (2014). *Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014*. CERN Open Data Portal. DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`

---

## Data Pipeline Stages

- `raw/`: Downloaded `atlas-higgs-challenge-2014-v2.csv.gz` directly from CERN Open Data (`186.5 MiB` compressed).
- `interim/`: Intermediate cleaned or partitioned datasets (`train`, `validation`, `test`, `holdout`).
- `processed/`: Encoded feature sets, imputed matrices, or precomputed summary statistics.

---

## Reproducing & Deleting Data

To download the official dataset:
```bash
python scripts/download_dataset.py
```

To validate and partition the dataset:
```bash
python scripts/validate_dataset.py
```

To wipe and reproduce cleanly from scratch:
```bash
rm -rf data/raw/* data/interim/* data/processed/*
python scripts/download_dataset.py
python scripts/validate_dataset.py
```
