from pathlib import Path

import pytest
from ml.data.prep_pipeline import EMPTY_HASH, DatasetPrepPipeline


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

    factors = manifest["weight_renormalization_factors"]
    assert set(factors.keys()) == {"train", "validation", "test", "holdout"}

    for name in ["train", "validation", "test", "holdout"]:
        assert name in factors
        assert "signal_factor" in factors[name]
        assert "background_factor" in factors[name]
        assert factors[name]["signal_factor"] > 0
        assert factors[name]["background_factor"] > 0

    for name, df in splits.items():
        if len(df) == 0:
            continue
        s_renorm = float(df.loc[df["Label"] == "s", "RenormalizedWeight"].sum())
        b_renorm = float(df.loc[df["Label"] == "b", "RenormalizedWeight"].sum())

        if (df["Label"] == "s").sum() > 0:
            assert pytest.approx(s_renorm, rel=1e-5) == s_full
        if (df["Label"] == "b").sum() > 0:
            assert pytest.approx(b_renorm, rel=1e-5) == b_full


def test_prep_pipeline_sentinel_strategies(mock_raw_df):
    p_keep = DatasetPrepPipeline(sentinel_strategy="keep-as-value")
    s_keep, _ = p_keep.process(mock_raw_df)
    assert (s_keep["train"]["PRI_jet_leading_pt"] == -999.0).any()

    p_mask = DatasetPrepPipeline(sentinel_strategy="mask")
    s_mask, _ = p_mask.process(mock_raw_df)
    assert s_mask["train"]["PRI_jet_leading_pt"].isna().any()
    assert not (s_mask["train"]["PRI_jet_leading_pt"] == -999.0).any()

    p_imp = DatasetPrepPipeline(sentinel_strategy="impute")
    s_imp, _ = p_imp.process(mock_raw_df)
    assert not s_imp["train"]["PRI_jet_leading_pt"].isna().any()
    assert not (s_imp["train"]["PRI_jet_leading_pt"] == -999.0).any()


def test_prep_pipeline_content_hash_validity_and_sensitivity(mock_raw_df, tmp_path: Path):
    pipeline = DatasetPrepPipeline(dataset_version="test_v1", seed=42)

    dir1 = tmp_path / "run1"
    _, m1 = pipeline.run_export(mock_raw_df, output_dir=dir1)

    hash1 = m1["content_hash"]
    assert hash1 != EMPTY_HASH
    assert len(hash1) == 64

    # Test idempotence
    dir2 = tmp_path / "run2"
    _, m2 = pipeline.run_export(mock_raw_df, output_dir=dir2)
    assert m2["content_hash"] == hash1

    # Test hash sensitivity on altered data
    mock_altered = mock_raw_df.copy()
    mock_altered.loc[0, "PRI_tau_pt"] += 999.0

    dir3 = tmp_path / "run3"
    _, m3 = pipeline.run_export(mock_altered, output_dir=dir3)
    assert m3["content_hash"] != hash1
    assert m3["content_hash"] != EMPTY_HASH
