import pandas as pd
import pytest

from backend.app.data.pipeline import DatasetPipeline


def test_official_partition_splits(mock_raw_df, monkeypatch):
    pipeline = DatasetPipeline()
    monkeypatch.setattr(pipeline, "load_raw", lambda: mock_raw_df)

    parts = pipeline.get_partition_splits(strategy="official_partition")
    assert "train" in parts
    assert "validation" in parts
    assert "test" in parts
    assert "holdout" in parts

    # Verify exact mapping
    assert (parts["train"]["KaggleSet"] == "t").all()
    assert (parts["validation"]["KaggleSet"] == "b").all()
    assert (parts["test"]["KaggleSet"] == "v").all()
    assert (parts["holdout"]["KaggleSet"] == "u").all()


def test_stratified_random_splits(mock_raw_df, monkeypatch):
    pipeline = DatasetPipeline()
    monkeypatch.setattr(pipeline, "load_raw", lambda: mock_raw_df)

    parts = pipeline.get_partition_splits(strategy="stratified_random", seed=123)
    assert len(parts["train"]) > 0
    assert len(parts["validation"]) > 0
    assert len(parts["test"]) > 0


def test_verify_no_overlap():
    p1 = pd.DataFrame({"EventId": [1, 2, 3]})
    p2 = pd.DataFrame({"EventId": [4, 5, 6]})
    DatasetPipeline.verify_no_overlap({"p1": p1, "p2": p2})

    # Test failure when overlap exists
    p3 = pd.DataFrame({"EventId": [3, 7, 8]})
    with pytest.raises(RuntimeError, match="Data Leakage Error"):
        DatasetPipeline.verify_no_overlap({"p1": p1, "p3": p3})
