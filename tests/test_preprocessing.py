import numpy as np

from backend.app.features.preprocessing import PreprocessingPipeline


def test_preprocessing_impute_median_scale(mock_raw_df):
    pipeline = PreprocessingPipeline(strategy="impute_median_scale")
    X_train = pipeline.fit_transform(mock_raw_df, feature_set="primary_only")

    assert pipeline.is_fitted
    assert not np.isnan(X_train).any()
    assert not (X_train == -999.0).any()

    # Test zero leakage: transform a new dataframe without re-fitting
    X_val = pipeline.transform(mock_raw_df)
    assert X_val.shape[1] == X_train.shape[1]


def test_preprocessing_native_strategy(mock_raw_df):
    pipeline = PreprocessingPipeline(strategy="native")
    X = pipeline.fit_transform(mock_raw_df, feature_set="all_physics")
    # Native should replace -999.0 with np.nan for tree models
    assert np.isnan(X).any()
    assert not (X == -999.0).any()
