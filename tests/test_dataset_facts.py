import gzip
import os

import pandas as pd
import pytest

from backend.app.schemas.dataset import DERIVED_FEATURES, METADATA_COLUMNS, PRIMARY_FEATURES


def test_schema_exact_counts():
    assert len(PRIMARY_FEATURES) == 17, f"Expected 17 primary features, found {len(PRIMARY_FEATURES)}"
    assert len(DERIVED_FEATURES) == 13, f"Expected 13 derived features, found {len(DERIVED_FEATURES)}"
    assert len(PRIMARY_FEATURES) + len(DERIVED_FEATURES) == 30, "Total physics features must be 30"
    assert len(METADATA_COLUMNS) == 5, f"Expected 5 metadata columns, found {len(METADATA_COLUMNS)}"
    assert len(PRIMARY_FEATURES) + len(DERIVED_FEATURES) + len(METADATA_COLUMNS) == 35, "Total schema columns must be 35"


def test_partition_exact_counts_and_sentinel():
    raw_path = "D:/HiggsLens/data/raw/atlas-higgs-challenge-2014-v2.csv.gz"
    if not os.path.exists(raw_path):
        pytest.skip("Raw CERN dataset not found at D:/HiggsLens/data/raw/atlas-higgs-challenge-2014-v2.csv.gz")

    df = pd.read_csv(raw_path)
    assert len(df) == 818238, f"Expected total 818,238 rows, found {len(df)}"

    counts = df["KaggleSet"].value_counts().to_dict()
    assert counts.get("t", 0) == 250000, "Training partition (t) must be exactly 250,000 rows"
    assert counts.get("b", 0) == 100000, "Validation partition (b) must be exactly 100,000 rows"
    assert counts.get("v", 0) == 450000, "Test partition (v) must be exactly 450,000 rows"
    assert counts.get("u", 0) == 18238, "Holdout partition (u) must be exactly 18,238 rows"
    assert counts["t"] + counts["b"] + counts["v"] + counts["u"] == 818238

    # Verify column count against schema
    assert len(df.columns) == 35, f"Expected 35 columns in raw CSV, found {len(df.columns)}"
