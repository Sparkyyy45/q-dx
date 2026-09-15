"""
Unit tests for feature reduction and feature selection methods.
"""

import numpy as np
import pandas as pd
import pytest

from src.reduction import FeatureReducer, select_features, reduce_features


def test_reduction_none():
    X = pd.DataFrame(np.random.randn(50, 6), columns=[f"f_{i}" for i in range(6)])
    y = np.random.binomial(1, 0.5, 50)
    X_red, art, red = select_features(X, y, strategy="none")
    assert X_red.shape == (50, 6)
    assert not art.is_latent


def test_reduction_f_classif():
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f_{i}" for i in range(10)])
    y = np.random.binomial(1, 0.5, 100)
    X_red, art, red = select_features(X, y, strategy="f_classif", k=4)
    assert X_red.shape == (100, 4)
    assert len(art.output_feature_names) == 4
    assert art.scores is not None


def test_reduction_pca_latent_notice():
    X = pd.DataFrame(np.random.randn(80, 8), columns=[f"f_{i}" for i in range(8)])
    y = np.random.binomial(1, 0.5, 80)
    X_red, art, red = select_features(X, y, strategy="pca", pca_components=3)
    assert X_red.shape == (80, 3)
    assert art.is_latent is True
    assert "latent" in art.clinical_notice.lower()
    assert art.output_feature_names == ["PC_1", "PC_2", "PC_3"]


def test_reduction_column_permutation_alignment():
    """Verify that permuted columns in evaluation DataFrame yield identical reduction output."""
    X_train = pd.DataFrame({
        "f_0": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "f_2": [100.0, 200.0, 300.0, 400.0, 500.0],
    })
    y_train = np.array([0, 0, 1, 1, 1])
    X_tr_red, art, red = select_features(X_train, y_train, strategy="f_classif", k=2)

    X_val_normal = pd.DataFrame({
        "f_0": [2.5],
        "f_1": [25.0],
        "f_2": [250.0],
    })
    X_val_permuted = pd.DataFrame({
        "f_2": [250.0],
        "f_0": [2.5],
        "f_1": [25.0],
    })

    red_normal = reduce_features(art, red, X_val_normal)
    red_permuted = reduce_features(art, red, X_val_permuted)
    assert np.allclose(red_normal.values, red_permuted.values)


def test_reduction_missing_column_error():
    """Verify that missing input column in FeatureReducer raises descriptive ValueError."""
    X_train = pd.DataFrame({
        "f_0": [1.0, 2.0, 3.0],
        "f_1": [10.0, 20.0, 30.0],
    })
    y_train = np.array([0, 1, 1])
    X_tr_red, art, red = select_features(X_train, y_train, strategy="none")

    X_val_missing = pd.DataFrame({"f_0": [2.5]})
    with pytest.raises(ValueError, match="missing required features"):
        red.transform(X_val_missing)

