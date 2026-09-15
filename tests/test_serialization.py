"""
Unit tests for production model pipeline serialization, artifact bundling,
and round-trip numerical invariance.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.features import ClinicalFeatureEngineer
from src.models import create_model
from src.models.serialization import (
    ProductionPipeline,
    load_production_pipeline,
    save_production_pipeline,
)
from src.preprocess import build_preprocessor, fit_preprocessor


def _get_synthetic_data() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame({
        "age": [18250, 20000, 19000, 21000, 18500, 22000, 19500, 20500],
        "gender": [1, 2, 1, 2, 1, 2, 1, 2],
        "height": [165, 175, 160, 180, 168, 172, 162, 178],
        "weight": [65, 80, 70, 85, 60, 90, 68, 82],
        "ap_hi": [120, 140, 130, 150, 110, 160, 125, 145],
        "ap_lo": [80, 90, 85, 95, 70, 100, 82, 92],
        "cholesterol": [1, 2, 1, 3, 1, 2, 1, 3],
        "gluc": [1, 1, 2, 1, 1, 2, 1, 1],
        "smoke": [0, 1, 0, 1, 0, 1, 0, 1],
        "alco": [0, 0, 1, 0, 0, 1, 0, 0],
        "active": [1, 1, 1, 0, 1, 0, 1, 1],
    })
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return df, y


def test_classical_pipeline_serialization_round_trip():
    """Verify that a fitted classical model pipeline can be saved, loaded, and matches predictions exactly."""
    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    model = create_model("logistic_regression", config=cfg)
    model.fit(df_fe, y)

    meta = {
        "model_key": "logistic_regression",
        "model_name": "Logistic Regression",
        "locked_threshold": 0.4920,
        "dataset_sha256": "dummy_sha256",
        "train_date": "2026-09-07T00:00:00Z",
    }
    pipe = ProductionPipeline(
        model=model,
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        metadata=meta,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "logistic_regression"
        save_production_pipeline(pipe, out_path)

        assert (out_path / "pipeline.joblib").is_file()
        assert (out_path / "metadata.json").is_file()

        loaded_pipe = load_production_pipeline(out_path)
        assert loaded_pipe.locked_threshold == 0.4920
        assert loaded_pipe.metadata["model_key"] == "logistic_regression"

        p_orig = pipe.predict_risk(df)
        p_load = loaded_pipe.predict_risk(df)
        assert np.allclose(p_orig, p_load, atol=1e-6)

        pred_orig = pipe.predict(df)
        pred_load = loaded_pipe.predict(df)
        assert np.array_equal(pred_orig, pred_load)


def test_quantum_vqc_serialization_round_trip():
    """Verify that a fitted VQC model pipeline preserves weights and predicts identically upon reload."""
    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    vqc = create_model("vqc", config=cfg)
    vqc.max_train_samples = 10
    vqc.n_epochs = 2
    vqc.fit(df_fe, y)

    meta = {
        "model_key": "vqc",
        "model_name": "Variational Quantum Classifier",
        "locked_threshold": 0.5150,
    }
    pipe = ProductionPipeline(
        model=vqc,
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        metadata=meta,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "vqc"
        save_production_pipeline(pipe, out_path)

        assert (out_path / "pipeline.joblib").is_file()
        assert (out_path / "metadata.json").is_file()
        assert (out_path / "circuit_config.json").is_file()

        with open(out_path / "circuit_config.json", "r") as f:
            circuit_cfg = json.load(f)
        assert circuit_cfg["n_qubits"] == 4

        loaded_pipe = load_production_pipeline(out_path)
        assert loaded_pipe.locked_threshold == 0.5150

        p_orig = pipe.predict_risk(df)
        p_load = loaded_pipe.predict_risk(df)
        assert np.allclose(p_orig, p_load, atol=1e-5)


def test_decoupled_artifacts_created_and_stateless_reconstruction():
    """
    ISSUE 14 Validation:
    Verify decoupled modular artifacts (preprocessing.joblib, model.joblib,
    threshold.json, schema.json, metadata.json) are created and can reconstruct
    the pipeline in a fresh context without pipeline.joblib.
    """
    from src.models.serialization import reconstruct_production_pipeline

    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    model = create_model("logistic_regression", config=cfg)
    model.fit(df_fe, y)

    meta = {
        "model_key": "logistic_regression",
        "model_name": "Logistic Regression",
        "locked_threshold": 0.4620,
        "features": list(df_fe.columns),
    }
    pipe = ProductionPipeline(
        model=model,
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        metadata=meta,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "logistic_regression_decoupled"
        save_production_pipeline(pipe, out_path)

        # Confirm all decoupled artifacts exist
        assert (out_path / "preprocessing.joblib").is_file()
        assert (out_path / "model.joblib").is_file()
        assert (out_path / "threshold.json").is_file()
        assert (out_path / "schema.json").is_file()
        assert (out_path / "metadata.json").is_file()

        # Delete monolithic pipeline.joblib to prove reconstruction is stateless
        (out_path / "pipeline.joblib").unlink()
        assert not (out_path / "pipeline.joblib").is_file()

        # Reconstruct purely from modular components
        reconstructed_pipe = reconstruct_production_pipeline(out_path)
        assert reconstructed_pipe.locked_threshold == 0.4620
        assert reconstructed_pipe.model_name == "Logistic Regression"

        p_orig = pipe.predict_risk(df)
        p_reconstructed = reconstructed_pipe.predict_risk(df)
        assert np.allclose(p_orig, p_reconstructed, atol=1e-7)

        pred_orig = pipe.predict(df)
        pred_reconstructed = reconstructed_pipe.predict(df)
        assert np.array_equal(pred_orig, pred_reconstructed)


def test_training_side_inference_preprocessing_parity():
    """
    ISSUE 4 Validation:
    Verify that the training-side preprocessing output for a raw patient
    is bitwise/numerically identical to production pipeline.transform_raw_dataframe().
    """
    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    from src.preprocess import transform_data
    # 1. Training-side transform
    test_patient_raw = df.iloc[[0]].copy()
    patient_fe_train = fe.transform(test_patient_raw).drop(columns=["age"])
    patient_proc_train = transform_data(prep_art, patient_fe_train)

    # 2. Production pipeline transform
    pipe = ProductionPipeline(
        model=create_model("logistic_regression", config=cfg),
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        metadata={"features": list(patient_proc_train.columns)},
    )
    patient_proc_prod = pipe.transform_raw_dataframe(test_patient_raw)

    # 3. Assert parity: column names, column order, and values must match exactly
    assert list(patient_proc_train.columns) == list(patient_proc_prod.columns)
    assert np.allclose(patient_proc_train.values, patient_proc_prod.values, atol=1e-7)


def test_tree_shap_local_explainability_in_pipeline():
    """
    ISSUE 10 Validation:
    Verify that ProductionPipeline.explain() produces genuine local TreeSHAP
    explanations for tree models.
    """
    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    rf = create_model("random_forest", config=cfg)
    rf.fit(df_fe, y)

    pipe = ProductionPipeline(
        model=rf,
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        metadata={"features": list(df_fe.columns), "locked_threshold": 0.50},
    )

    exp = pipe.explain(df.iloc[[0]])
    assert exp["explanation_method"] == "treeshap"
    assert exp["method"] == "treeshap"
    assert len(exp["top_factors"]) > 0
    assert "attribution" in exp["top_factors"][0]
    assert isinstance(exp["top_factors"][0]["attribution"], float)
    assert exp["threshold_source"] == "OOF_Youden"


def test_catboost_champion_pipeline_parity_and_stateless_reconstruction():
    """
    P0-5, P0-6, P0-9, P0-10 Validation:
    Verify CatBoost production champion pipeline:
    1. Fits with native categorical bypass preprocessor.
    2. Serializes decoupled modular artifacts.
    3. Reconstructs statelessly in a clean directory without pipeline.joblib.
    4. Guarantees bitwise/numerical prediction parity.
    5. Produces genuine local TreeSHAP attributions with 'method': 'treeshap'.
    """
    from src.models.serialization import reconstruct_production_pipeline
    from src.preprocess import build_catboost_preprocessor

    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_catboost_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    cb_model = create_model("catboost", config=cfg)
    cb_model.fit(df_fe, y)

    meta = {
        "model_key": "catboost",
        "model_name": "CatBoost",
        "locked_threshold": 0.4836,
        "features": list(df_fe.columns),
        "is_catboost": True,
    }
    pipe = ProductionPipeline(
        model=cb_model,
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        preprocessor=prep,
        metadata=meta,
        is_catboost=True,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "catboost_champion"
        save_production_pipeline(pipe, out_path)

        # Confirm all decoupled modular components are written
        assert (out_path / "preprocessing.joblib").is_file()
        assert (out_path / "model.joblib").is_file()
        assert (out_path / "threshold.json").is_file()
        assert (out_path / "schema.json").is_file()
        assert (out_path / "metadata.json").is_file()

        # Delete monolithic blob
        (out_path / "pipeline.joblib").unlink()

        # Reconstruct statelessly
        reconstructed = reconstruct_production_pipeline(out_path)
        assert reconstructed.locked_threshold == 0.4836
        assert reconstructed.model_name == "CatBoost"
        assert reconstructed.is_catboost is True

        # Assert prediction parity on unseen test patient
        test_patient = df.iloc[[1]]
        p_orig = pipe.predict_risk(test_patient)
        p_recon = reconstructed.predict_risk(test_patient)
        assert np.allclose(p_orig, p_recon, atol=1e-7)

        pred_orig = pipe.predict(test_patient)
        pred_recon = reconstructed.predict(test_patient)
        assert np.array_equal(pred_orig, pred_recon)

        # Assert genuine TreeSHAP explainability
        exp = reconstructed.explain(test_patient)
        assert exp["method"] == "treeshap"
        assert exp["explanation_method"] == "treeshap"
        assert exp["applied_threshold"] == 0.4836
        assert len(exp["top_factors"]) > 0
        assert "contribution_direction" in exp["top_factors"][0]
        assert "feature_label" in exp["top_factors"][0]


def test_catboost_treeshap_additivity_in_margin_log_odds_space():
    """
    P0-3 Validation:
    Verify TreeSHAP additivity in the correct output space for CatBoost:
    Margin output = base_value + sum(shap_values) to machine precision.
    Sigmoid(margin_output) equals calibrated risk probability.
    """
    import shap
    from src.preprocess import build_catboost_preprocessor

    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_catboost_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    cb_model = create_model("catboost", config=cfg)
    cb_model.fit(df_fe, y)

    pipe = ProductionPipeline(
        model=cb_model,
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        preprocessor=prep,
        metadata={"model_name": "CatBoost", "locked_threshold": 0.50},
        is_catboost=True,
    )

    test_patient = df.iloc[[0]]
    exp = pipe.explain(test_patient)
    assert exp["explanation_method"] == "treeshap"
    assert exp["explanation_available"] is True
    assert exp["shap_output_space"] == "margin_log_odds"

    # Compute raw margin and shap values directly on underlying model
    df_ready = pipe.transform_raw_dataframe(test_patient)
    explainer = shap.TreeExplainer(cb_model.model)
    res = explainer(df_ready)
    base_val = float(res.base_values[0])
    shap_sum = float(res.values[0].sum())
    margin = float(cb_model.model.predict(df_ready, prediction_type="RawFormulaVal")[0])
    prob = float(pipe.predict_risk(test_patient)[0])

    # TreeSHAP exact additivity in margin log-odds space
    assert np.isclose(base_val + shap_sum, margin, atol=1e-5)
    # Sigmoid link between margin and probability
    expected_prob = 1.0 / (1.0 + np.exp(-margin))
    assert np.isclose(prob, expected_prob, atol=1e-5)


def test_treeshap_failure_returns_treeshap_unavailable_without_silent_deviation():
    """
    P0-3 Validation:
    Verify that if TreeSHAP fails or raises an exception, the system returns
    explanation_available: False and method: 'treeshap_unavailable',
    and NEVER silently disguises feature deviation as SHAP.
    """
    from unittest.mock import patch
    import shap
    from src.preprocess import build_catboost_preprocessor

    df, y = _get_synthetic_data()
    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer().fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    prep = build_catboost_preprocessor(cfg, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, df_fe, y, num_cols, cat_cols, cfg)

    cb_model = create_model("catboost", config=cfg)
    cb_model.fit(df_fe, y)

    pipe = ProductionPipeline(
        model=cb_model,
        feature_engineer=fe,
        preprocessor_artifact=prep_art,
        preprocessor=prep,
        metadata={"model_name": "CatBoost", "locked_threshold": 0.50},
        is_catboost=True,
    )

    test_patient = df.iloc[[0]]

    # Force TreeExplainer to raise an exception
    with patch.object(shap, "TreeExplainer", side_effect=RuntimeError("Simulated TreeSHAP C++ runtime memory failure")):
        exp = pipe.explain(test_patient)

    assert exp["explanation_available"] is False
    assert exp["method"] == "treeshap_unavailable"
    assert exp["explanation_method"] == "treeshap_unavailable"
    assert exp["top_factors"] == []


