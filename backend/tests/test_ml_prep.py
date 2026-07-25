import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml.data.prep_pipeline import DatasetPrepPipeline


def test_prep_pipeline_kaggleset_partition_mapping(mock_raw_df):
    pipeline = DatasetPrepPipeline(sentinel_strategy="keep-as-value", seed=42)
    splits, manifest = pipeline.process(mock_raw_df)

    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    assert "holdout" in splits

    assert (splits["train"]["KaggleSet"] == "t").all()
    assert (splits["validation"]["KaggleSet"] == "b").all()
    assert (splits["test"]["KaggleSet"] == "v").all()
    assert (splits["holdout"]["KaggleSet"] == "u").all()

    assert manifest["canonical_split_mapping"]["t"] == "train"
    assert manifest["canonical_split_mapping"]["b"] == "validation"


def test_prep_pipeline_preserves_weight_label_columns(mock_raw_df):
    pipeline = DatasetPrepPipeline()
    splits, _ = pipeline.process(mock_raw_df)

    for name, df in splits.items():
        assert "EventId" in df.columns
        assert "Label" in df.columns
        assert "Weight" in df.columns
        assert "RenormalizedWeight" in df.columns
        assert "KaggleSet" in df.columns


def test_prep_pipeline_luminosity_weight_renormalization(mock_raw_df):
    pipeline = DatasetPrepPipeline()
    splits, manifest = pipeline.process(mock_raw_df)

    s_full = float(mock_raw_df.loc[mock_raw_df["Label"] == "s", "Weight"].sum())
    b_full = float(mock_raw_df.loc[mock_raw_df["Label"] == "b", "Weight"].sum())

    for name, df in splits.items():
        s_renorm = float(df.loc[df["Label"] == "s", "RenormalizedWeight"].sum())
        b_renorm = float(df.loc[df["Label"] == "b", "RenormalizedWeight"].sum())

        if (df["Label"] == "s").sum() > 0:
            assert pytest.approx(s_renorm, rel=1e-5) == s_full
        if (df["Label"] == "b").sum() > 0:
            assert pytest.approx(b_renorm, rel=1e-5) == b_full


def test_prep_pipeline_sentinel_strategies(mock_raw_df):
    # Strategy 1: keep-as-value
    p_keep = DatasetPrepPipeline(sentinel_strategy="keep-as-value")
    s_keep, _ = p_keep.process(mock_raw_df)
    assert (s_keep["train"]["PRI_jet_leading_pt"] == -999.0).any()

    # Strategy 2: mask
    p_mask = DatasetPrepPipeline(sentinel_strategy="mask")
    s_mask, _ = p_mask.process(mock_raw_df)
    assert s_mask["train"]["PRI_jet_leading_pt"].isna().any()
    assert not (s_mask["train"]["PRI_jet_leading_pt"] == -999.0).any()

    # Strategy 3: impute
    p_imp = DatasetPrepPipeline(sentinel_strategy="impute")
    s_imp, _ = p_imp.process(mock_raw_df)
    assert not s_imp["train"]["PRI_jet_leading_pt"].isna().any()
    assert not (s_imp["train"]["PRI_jet_leading_pt"] == -999.0).any()


def test_prep_pipeline_idempotence_and_content_hash(mock_raw_df, tmp_path: Path):
    pipeline = DatasetPrepPipeline(dataset_version="test_v1", seed=42)

    dir1 = tmp_path / "run1"
    _, m1 = pipeline.run_export(mock_raw_df, output_dir=dir1)

    dir2 = tmp_path / "run2"
    _, m2 = pipeline.run_export(mock_raw_df, output_dir=dir2)

    assert m1["content_hash"] == m2["content_hash"]
    assert m1["total_events"] == m2["total_events"]
