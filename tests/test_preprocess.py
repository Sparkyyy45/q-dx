"""
Unit tests for preprocessing, scalers, imputers, and IQR outlier clipping.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.preprocess import (
    IQRClipper,
    build_preprocessor,
    fit_preprocessor,
    prepare_fold,
    transform_data,
)


def test_iqr_clipper_bounds():
    # 10 values: 1 to 10
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    clipper = IQRClipper(factor=1.5)
    clipper.fit(X_train)

    q1 = np.percentile(X_train, 25)
    q3 = np.percentile(X_train, 75)
    iqr = q3 - q1
    expected_low = q1 - 1.5 * iqr
    expected_high = q3 + 1.5 * iqr

    assert np.isclose(clipper.lower_bounds_[0], expected_low)
    assert np.isclose(clipper.upper_bounds_[0], expected_high)

    # Test that extreme values in test data are clipped to learned bounds
    X_test = np.array([[-100.0], [100.0], [5.0]])
    clipped = clipper.transform(X_test)
    assert np.isclose(clipped[0, 0], expected_low)
    assert np.isclose(clipped[1, 0], expected_high)
    assert np.isclose(clipped[2, 0], 5.0)


def test_prepare_fold_pipeline():
    X_train = pd.DataFrame({
        "num_1": [10.0, 20.0, np.nan, 40.0, 50.0],
        "cat_1": ["A", "B", "A", "C", "B"],
    })
    y_train = np.array([0, 1, 0, 1, 0])

    X_val = pd.DataFrame({
        "num_1": [15.0, 999.0],  # Outlier to be clipped
        "cat_1": ["A", "UNKNOWN_CATEGORY"],  # Unknown category to be ignored
    })

    config = PipelineConfig(scaling_strategy="standard", outlier_clipping=True)
    X_tr_proc, X_val_proc, artifact = prepare_fold(
        X_train, y_train, X_val, config, ["num_1"], ["cat_1"]
    )

    assert X_tr_proc.shape[0] == 5
    assert X_val_proc.shape[0] == 2
    assert not X_tr_proc.isnull().any().any()
    assert not X_val_proc.isnull().any().any()
    assert len(artifact.output_feature_names) > 0


def test_scaling_strategies():
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0]})
    for strat in ["standard", "robust", "minmax", "none"]:
        cfg = PipelineConfig(scaling_strategy=strat, outlier_clipping=False)
        prep = build_preprocessor(cfg, ["num"], [])
        art = fit_preprocessor(prep, X, None, ["num"], [], cfg)
        out = transform_data(art, X)
        assert out.shape == (5, 1)
        if strat == "minmax":
            assert np.isclose(out.min().iloc[0], 0.0)
            assert np.isclose(out.max().iloc[0], 1.0)
