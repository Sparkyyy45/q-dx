"""
Unit tests verifying uniform interface and output semantics for all five ML models.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.models import create_model, get_all_models
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm_calibrated import CalibratedSVMModel


def test_all_eight_models_uniform_interface():
    config = PipelineConfig()
    models = get_all_models(config)
    assert len(models) == 8

    X_train = pd.DataFrame(np.random.randn(80, 5), columns=[f"col_{i}" for i in range(5)])
    y_train = np.random.binomial(1, 0.5, 80)
    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f"col_{i}" for i in range(5)])
    y_test = np.random.binomial(1, 0.5, 20)

    for m in models:
        # 1. Fit
        m.fit(X_train, y_train)
        assert m.is_fitted_

        # 2. Predict
        preds = m.predict(X_test)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1})

        # 3. Predict proba
        probas = m.predict_proba(X_test)
        assert probas.shape == (20, 2)
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
        assert np.allclose(probas.sum(axis=1), 1.0)

        # 4. Predict risk
        risk = m.predict_risk(X_test)
        assert risk.shape == (20,)
        assert np.all(risk >= 0.0) and np.all(risk <= 1.0)

        # 5. Evaluate
        metrics = m.evaluate(X_test, y_test)
        assert "roc_auc" in metrics
        assert "accuracy" in metrics
        assert "brier_score" in metrics


def test_xgboost_model():
    from src.models.xgboost_model import XGBoostModel
    config = PipelineConfig()
    xgb = XGBoostModel(config)
    assert xgb.name == "XGBoost"

    X = pd.DataFrame(np.random.randn(50, 4), columns=["a", "b", "c", "d"])
    y = np.random.binomial(1, 0.5, 50)
    xgb.fit(X, y)
    assert xgb.is_fitted_

    preds = xgb.predict(X)
    assert len(preds) == 50
    prob = xgb.predict_risk(X)
    assert len(prob) == 50
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_lightgbm_model():
    from src.models.lightgbm_model import LightGBMModel
    config = PipelineConfig()
    lgb = LightGBMModel(config)
    assert lgb.name == "LightGBM"

    X = pd.DataFrame(np.random.randn(50, 4), columns=["a", "b", "c", "d"])
    y = np.random.binomial(1, 0.5, 50)
    lgb.fit(X, y)
    assert lgb.is_fitted_

    preds = lgb.predict(X)
    assert len(preds) == 50
    prob = lgb.predict_risk(X)
    assert len(prob) == 50
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_catboost_model():
    from src.models.catboost_model import CatBoostModel
    config = PipelineConfig()
    cb = CatBoostModel(config)
    assert cb.name == "CatBoost"

    X = pd.DataFrame({
        "ap_hi": np.random.randn(50),
        "gender": np.random.choice([1, 2], size=50),
        "cholesterol": np.random.choice([1, 2, 3], size=50),
        "gluc": np.random.choice([1, 2, 3], size=50),
    })
    y = np.random.binomial(1, 0.5, 50)
    cb.fit(X, y)
    assert cb.is_fitted_

    preds = cb.predict(X)
    assert len(preds) == 50
    prob = cb.predict_risk(X)
    assert len(prob) == 50
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_catboost_categorical_columns_bypass_scaling_and_passed_as_cat_features():
    from src.preprocess import build_catboost_preprocessor, fit_preprocessor, transform_data
    from src.models.catboost_model import CatBoostModel

    config = PipelineConfig(scaling_strategy="robust")
    num_cols = ["ap_hi", "ap_lo", "gender", "cholesterol", "gluc"]
    cat_cols = []

    df = pd.DataFrame({
        "ap_hi": [120.0, 140.0, 160.0, 110.0, 130.0],
        "ap_lo": [80.0, 90.0, 100.0, 70.0, 85.0],
        "gender": [1, 2, 1, 2, 1],
        "cholesterol": [1, 2, 3, 1, 2],
        "gluc": [1, 1, 2, 3, 1],
    })
    y = np.array([0, 1, 1, 0, 1])

    # Build CatBoost preprocessor
    prep = build_catboost_preprocessor(config, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df, y, num_cols, cat_cols, config)
    df_proc = transform_data(prep_art, df)

    # 1. Verify categorical columns bypass RobustScaler (values remain unchanged discrete integers)
    assert list(df_proc["gender"].values) == [1.0, 2.0, 1.0, 2.0, 1.0]
    assert list(df_proc["cholesterol"].values) == [1.0, 2.0, 3.0, 1.0, 2.0]
    assert list(df_proc["gluc"].values) == [1.0, 1.0, 2.0, 3.0, 1.0]

    # 2. Verify pure numeric columns ARE scaled (transformed away from raw values)
    assert not np.allclose(df_proc["ap_hi"].values, df["ap_hi"].values)
    assert not np.allclose(df_proc["ap_lo"].values, df["ap_lo"].values)

    # 3. Fit CatBoostModel on processed data and verify cat_features are recognized
    cb_model = CatBoostModel(config)
    cb_model.fit(df_proc, y)
    assert cb_model.is_fitted_
    assert set(cb_model.cat_feature_names_) == {"gender", "cholesterol", "gluc"}
    assert len(cb_model.cat_feature_indices_) == 3

    # 4. Predict works correctly
    preds = cb_model.predict(df_proc)
    assert len(preds) == len(df)
    probas = cb_model.predict_proba(df_proc)
    assert probas.shape == (len(df), 2)


def test_logistic_regression_odds_ratios():
    config = PipelineConfig()
    lr = LogisticRegressionModel(config)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    y = np.random.binomial(1, 0.5, 50)
    lr.fit(X, y)

    odds_ratios = lr.get_odds_ratios()
    assert len(odds_ratios) == 3
    for k, v in odds_ratios.items():
        assert v > 0.0  # Odds ratio is exp(beta), strictly positive


def test_calibrated_svm_probabilities():
    config = PipelineConfig()
    svm = CalibratedSVMModel(config)
    X = pd.DataFrame(np.random.randn(60, 4), columns=[f"f_{i}" for i in range(4)])
    y = np.random.binomial(1, 0.5, 60)
    svm.fit(X, y)

    proba = svm.predict_proba(X)
    assert proba.shape == (60, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_vif_computation():
    from src.explain_risk import compute_vif, explain_logistic_regression

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 3), columns=["x1", "x2", "x3"])
    # x4 is almost perfectly collinear with x1 + x2
    X["x4"] = X["x1"] * 0.95 + X["x2"] * 0.95 + np.random.randn(200) * 0.01
    y = np.random.binomial(1, 0.5, 200)

    vif_df = compute_vif(X)
    assert len(vif_df) == 4
    # The collinear pair must have high VIF (> 10)
    assert vif_df["VIF"].max() > 10.0
    assert any("Severe" in s for s in vif_df["Collinearity Severity"])

    # Test de-collinearized explanation
    lr = LogisticRegressionModel()
    lr.fit(X, y)
    clean_odds_df, full_vif_df = explain_logistic_regression(lr, X_train=X, y_train=y)

    assert not clean_odds_df.empty
    # Every feature in clean_odds_df must have VIF <= 10
    assert (clean_odds_df["VIF"] <= 10.0).all()
    assert len(clean_odds_df) < len(X.columns)  # Collinear feature was pruned


def test_dataframe_column_permutation_alignment():
    """Verify that permuting column order in evaluation DataFrame does not scramble features."""
    config = PipelineConfig()
    lr = LogisticRegressionModel(config)
    X_train = pd.DataFrame({
        "col_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "col_b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    y_train = np.array([0, 0, 1, 1, 1])
    lr.fit(X_train, y_train)

    X_test_normal = pd.DataFrame({
        "col_a": [2.5, 3.5],
        "col_b": [25.0, 35.0],
    })
    X_test_permuted = pd.DataFrame({
        "col_b": [25.0, 35.0],
        "col_a": [2.5, 3.5],
    })

    prob_normal = lr.predict_risk(X_test_normal)
    prob_permuted = lr.predict_risk(X_test_permuted)
    assert np.allclose(prob_normal, prob_permuted), "Column order permutation scrambled model features!"


def test_missing_column_raises_clear_error():
    """Verify that passing DataFrame with missing required features raises clear ValueError."""
    config = PipelineConfig()
    lr = LogisticRegressionModel(config)
    X_train = pd.DataFrame({
        "col_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "col_b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    y_train = np.array([0, 0, 1, 1, 1])
    lr.fit(X_train, y_train)

    X_test_missing = pd.DataFrame({"col_a": [2.5, 3.5]})
    with pytest.raises(ValueError, match="missing required features"):
        lr.predict(X_test_missing)

