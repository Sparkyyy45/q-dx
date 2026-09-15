"""
Unit tests for clinical evaluation metrics and optimal threshold recalibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    EvaluationMetrics,
    create_threshold_recalibration_table,
    evaluate_predictions,
    find_optimal_threshold,
)


def test_optimal_threshold_youden():
    """
    Verify find_optimal_threshold recovers known hand-computable optimal Youden's J threshold.

    Hand-computed calculation:
      y_true = [0, 0, 0, 0, 1, 1, 1, 1]
      y_prob = [0.1, 0.2, 0.25, 0.6, 0.3, 0.5, 0.7, 0.8]
      Positives: [0.3, 0.5, 0.7, 0.8]
      Negatives: [0.1, 0.2, 0.25, 0.6]

      Threshold sweep from ROC curve:
      - Cutoff 0.8:  TP=1 (TPR=0.25), FP=0 (FPR=0.00) -> J = 0.25
      - Cutoff 0.7:  TP=2 (TPR=0.50), FP=0 (FPR=0.00) -> J = 0.50
      - Cutoff 0.6:  TP=2 (TPR=0.50), FP=1 (FPR=0.25) -> J = 0.25
      - Cutoff 0.5:  TP=3 (TPR=0.75), FP=1 (FPR=0.25) -> J = 0.50
      - Cutoff 0.3:  TP=4 (TPR=1.00), FP=1 (FPR=0.25) -> J = 0.75 (MAXIMUM)
      - Cutoff 0.25: TP=4 (TPR=1.00), FP=2 (FPR=0.50) -> J = 0.50
      - Cutoff 0.20: TP=4 (TPR=1.00), FP=3 (FPR=0.75) -> J = 0.25
      - Cutoff 0.10: TP=4 (TPR=1.00), FP=4 (FPR=1.00) -> J = 0.00

    Expected optimum: threshold = 0.3, sensitivity = 1.0, specificity = 0.75, youden_j = 0.75.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.25, 0.6, 0.3, 0.5, 0.7, 0.8])

    res = find_optimal_threshold(y_true, y_prob, method="youden")

    assert np.isclose(res["threshold"], 0.3, atol=1e-5), f"Expected threshold 0.3, got {res['threshold']}"
    assert np.isclose(res["sensitivity"], 1.0, atol=1e-5), f"Expected sensitivity 1.0, got {res['sensitivity']}"
    assert np.isclose(res["specificity"], 0.75, atol=1e-5), f"Expected specificity 0.75, got {res['specificity']}"
    assert np.isclose(res["youden_j"], 0.75, atol=1e-5), f"Expected Youden's J 0.75, got {res['youden_j']}"
    assert "cost_weighted_score" in res


def test_optimal_threshold_cost_weighted():
    """
    Verify asymmetric cost weighting shifts threshold in expected clinical direction:
    - Higher fn_cost (missing CVD is worse) lowers the threshold to increase sensitivity.
    - Higher fp_cost (false alarms are worse) raises the threshold to increase specificity.
    """
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.45, 0.4, 0.8])

    # Candidate cutoffs:
    # 0.8: TPR=0.5, FPR=0.0
    # 0.4: TPR=1.0, FPR=0.5

    # Case A: Missing positives is 3x more costly than false alarms
    # At 0.8: score = 0.5 * 3.0 - 0.0 * 1.0 = 1.5
    # At 0.4: score = 1.0 * 3.0 - 0.5 * 1.0 = 2.5 (optimal)
    res_high_fn = find_optimal_threshold(y_true, y_prob, method="cost_weighted", fn_cost=3.0, fp_cost=1.0)

    # Case B: False alarms are 3x more costly than missing positives
    # At 0.8: score = 0.5 * 1.0 - 0.0 * 3.0 = 0.5 (optimal)
    # At 0.4: score = 1.0 * 1.0 - 0.5 * 3.0 = -0.5
    res_high_fp = find_optimal_threshold(y_true, y_prob, method="cost_weighted", fn_cost=1.0, fp_cost=3.0)

    assert np.isclose(res_high_fn["threshold"], 0.4, atol=1e-5)
    assert np.isclose(res_high_fp["threshold"], 0.8, atol=1e-5)

    # Invariant: Higher fn_cost MUST strictly lower the optimal threshold
    assert res_high_fn["threshold"] < res_high_fp["threshold"], (
        f"Expected high fn_cost threshold ({res_high_fn['threshold']}) < high fp_cost threshold ({res_high_fp['threshold']})"
    )


def test_optimal_threshold_invalid_method():
    """Verify ValueError is raised for unsupported optimization methods."""
    with pytest.raises(ValueError):
        find_optimal_threshold(np.array([0, 1]), np.array([0.2, 0.8]), method="invalid_method")


def test_threshold_recalibration_table_generation():
    """Verify create_threshold_recalibration_table builds valid side-by-side rows."""
    mock_int_m = EvaluationMetrics(
        roc_auc=0.795, pr_auc=0.780, accuracy=0.73, balanced_accuracy=0.73,
        sensitivity=0.6729, specificity=0.7850, precision=0.75, npv=0.71, f1=0.71,
        brier_score=0.18, expected_calibration_error=0.01,
        confusion_matrix=[[78, 22], [33, 67]], tn=78, fp=22, fn=33, tp=67,
    )
    mock_ext_m = EvaluationMetrics(
        roc_auc=0.665, pr_auc=0.240, accuracy=0.52, balanced_accuracy=0.62,
        sensitivity=0.7711, specificity=0.4817, precision=0.20, npv=0.92, f1=0.32,
        brier_score=0.32, expected_calibration_error=0.15,
        confusion_matrix=[[48, 52], [23, 77]], tn=48, fp=52, fn=23, tp=77,
    )

    int_thresh = {
        "Logistic Regression": {
            "threshold": 0.4851, "sensitivity": 0.6950, "specificity": 0.7710, "youden_j": 0.4660, "cost_weighted_score": 0.4660
        }
    }
    ext_thresh = {
        "Logistic Regression": {
            "threshold": 0.1983, "sensitivity": 0.6387, "specificity": 0.6214, "youden_j": 0.2601, "cost_weighted_score": 0.2601
        }
    }

    df = create_threshold_recalibration_table(
        internal_metrics={"Logistic Regression": mock_int_m},
        internal_thresholds=int_thresh,
        external_metrics={"Logistic Regression": mock_ext_m},
        external_thresholds=ext_thresh,
        internal_prevalence=0.4938,
        external_prevalence=0.1498,
    )

    assert len(df) == 2
    assert "Model" in df.columns
    assert "Cohort" in df.columns
    assert "Fixed Cutoff" in df.columns
    assert "Optimal Cutoff" in df.columns
    assert "Optimal Sens" in df.columns
    assert "Optimal Spec" in df.columns
    assert "Youden's J" in df.columns

    int_row = df.iloc[0]
    ext_row = df.iloc[1]
    assert int_row["Fixed Sens"] == "0.6729"
    assert int_row["Optimal Cutoff"] == "0.4851"
    assert ext_row["Fixed Sens"] == "0.7711"
    assert ext_row["Optimal Cutoff"] == "0.1983"


def test_evaluate_at_locked_threshold_and_bootstrap_ci():
    """Verify evaluate_at_locked_threshold applies threshold strictly and generates 95% bootstrap CIs."""
    from src.evaluate import evaluate_at_locked_threshold, compute_bootstrap_ci

    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)
    # Give positives slightly higher scores
    y_prob = np.concatenate([np.random.uniform(0.1, 0.6, 50), np.random.uniform(0.4, 0.9, 50)])

    locked_tau = 0.52
    metrics = evaluate_at_locked_threshold(
        y_true, y_prob, locked_threshold=locked_tau, compute_ci=True, n_bootstraps=200, seed=42
    )

    assert metrics.applied_threshold == locked_tau
    assert metrics.bootstrap_ci is not None
    assert "roc_auc" in metrics.bootstrap_ci
    assert "sensitivity" in metrics.bootstrap_ci
    assert "specificity" in metrics.bootstrap_ci

    roc_ci = metrics.bootstrap_ci["roc_auc"]
    assert roc_ci["ci_lower"] <= roc_ci["mean"] <= roc_ci["ci_upper"]
    assert 0.5 <= roc_ci["mean"] <= 1.0


def test_oof_store_get_optimal_threshold():
    """Verify OutOfFoldStore.get_optimal_threshold locks Youden J threshold strictly from OOF predictions."""
    from src.evaluate import OutOfFoldStore

    oof = OutOfFoldStore(n_samples=8)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    oof.register_labels(np.arange(8), y_true)

    # Perfect separation at 0.5
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)

    oof.record_fold("MockModel", np.arange(8), y_true, y_pred, y_prob)

    opt = oof.get_optimal_threshold("MockModel", method="youden")
    assert "threshold" in opt
    assert "sensitivity" in opt
    assert "specificity" in opt
    assert "youden_j" in opt
    assert opt["sensitivity"] == 1.0
    assert opt["specificity"] == 1.0
    assert opt["youden_j"] == 1.0


def test_holdout_and_external_labels_cannot_influence_threshold_selection():
    """
    P0-3 Requirement:
    Verify that holdout labels and external cohort labels cannot influence threshold selection.
    Threshold is determined strictly from development OOF predictions and frozen.
    """
    from src.evaluate import OutOfFoldStore, evaluate_at_locked_threshold

    # 1. Development set OOF predictions
    n_dev = 100
    np.random.seed(42)
    y_dev_true = np.random.binomial(1, 0.5, size=n_dev)
    y_dev_prob = np.clip(y_dev_true * 0.4 + np.random.uniform(0.1, 0.5, size=n_dev), 0.0, 1.0)
    y_dev_pred = (y_dev_prob >= 0.5).astype(int)

    oof = OutOfFoldStore(n_samples=n_dev)
    oof.register_labels(np.arange(n_dev), y_dev_true)
    oof.record_fold("ChampionModel", np.arange(n_dev), y_dev_true, y_dev_pred, y_dev_prob)

    # Derive baseline locked threshold strictly from development OOF
    baseline_threshold = oof.get_optimal_threshold("ChampionModel", method="youden")["threshold"]

    # 2. Simulate Holdout test set with arbitrary labels
    n_holdout = 50
    y_holdout_a = np.ones(n_holdout, dtype=int)
    y_holdout_b = np.zeros(n_holdout, dtype=int)
    y_holdout_c = np.random.binomial(1, 0.2, size=n_holdout)
    prob_holdout = np.random.uniform(0.0, 1.0, size=n_holdout)

    # 3. Simulate External cohort with arbitrary labels
    n_external = 80
    y_external_skewed = np.random.binomial(1, 0.15, size=n_external)
    prob_external = np.random.uniform(0.0, 1.0, size=n_external)

    # Evaluating holdout sets at baseline_threshold must strictly use baseline_threshold
    m_h_a = evaluate_at_locked_threshold(y_holdout_a, prob_holdout, locked_threshold=baseline_threshold)
    m_h_b = evaluate_at_locked_threshold(y_holdout_b, prob_holdout, locked_threshold=baseline_threshold)
    m_h_c = evaluate_at_locked_threshold(y_holdout_c, prob_holdout, locked_threshold=baseline_threshold)
    assert m_h_a.applied_threshold == baseline_threshold
    assert m_h_b.applied_threshold == baseline_threshold
    assert m_h_c.applied_threshold == baseline_threshold

    # Evaluating external set at baseline_threshold must strictly use baseline_threshold
    m_ext = evaluate_at_locked_threshold(y_external_skewed, prob_external, locked_threshold=baseline_threshold)
    assert m_ext.applied_threshold == baseline_threshold

    # OutOfFoldStore threshold remains bit-for-bit identical (zero holdout/external contamination)
    frozen_threshold = oof.get_optimal_threshold("ChampionModel", method="youden")["threshold"]
    assert frozen_threshold == baseline_threshold


def test_select_champion_model_from_oof():
    """
    Verify champion model selection is strictly determined by development OOF metrics:
    development/OOF -> model selection -> champion chosen -> threshold lock -> holdout evaluation.
    """
    from src.evaluate import OutOfFoldStore

    n_dev = 200
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, size=n_dev)

    oof = OutOfFoldStore(n_samples=n_dev)
    oof.register_labels(np.arange(n_dev), y_true)

    # Model A: High ROC-AUC (0.85)
    prob_a = np.clip(y_true * 0.7 + np.random.uniform(0.0, 0.3, size=n_dev), 0.0, 1.0)
    pred_a = (prob_a >= 0.5).astype(int)
    oof.record_fold("ModelA_HighAUC", np.arange(n_dev), y_true, pred_a, prob_a)

    # Model B: Low ROC-AUC (0.65)
    prob_b = np.clip(y_true * 0.3 + np.random.uniform(0.1, 0.6, size=n_dev), 0.0, 1.0)
    pred_b = (prob_b >= 0.5).astype(int)
    oof.record_fold("ModelB_LowAUC", np.arange(n_dev), y_true, pred_b, prob_b)

    # Selection strictly selects Model A based on OOF ROC-AUC
    champ_name, champ_metrics = oof.select_champion_model(primary_metric="roc_auc")
    assert champ_name == "ModelA_HighAUC"
    assert champ_metrics.roc_auc > 0.80



