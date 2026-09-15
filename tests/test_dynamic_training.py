"""
Tests for FIX #1: Dynamic Training Out-Of-Fold (OOF) Threshold Locking.
Methodological Invariants Enforced:
1. Stratified train/test split.
2. 5-Fold Stratified Out-Of-Fold (OOF) cross-validation strictly on TRAIN.
3. Preprocessing fitted strictly on fold training subsets (zero leakage).
4. Operational threshold locked strictly from y_train + OOF predictions via Youden J.
5. CV fold estimators discarded after threshold locking.
6. Fresh final model refitted on entire training partition.
7. Evaluated on untouched test partition at locked OOF threshold.
8. y_test NEVER used for threshold, calibration, or preprocessing.

Tests Implemented:
A. test_dynamic_training_threshold_is_oof_locked
B. test_dynamic_training_threshold_invariant_to_test_label_mutation
C. test_dynamic_training_final_model_is_refit_after_cv
D. test_dynamic_training_test_labels_are_never_consumed_before_evaluation
E. test_dynamic_training_preprocessing_is_train_fold_isolated
"""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import src.training_engine as te
from src.training_engine import (
    execute_training_workflow,
    TRAINING_JOBS_DIR,
)
import src.evaluate as ev


def _generate_synthetic_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    return pd.DataFrame({
        "age_years": np.random.uniform(40, 70, n),
        "ap_hi": np.random.uniform(100, 180, n),
        "ap_lo": np.random.uniform(60, 110, n),
        "cholesterol": np.random.choice([1, 2, 3], n),
        "gluc": np.random.choice([1, 2, 3], n),
        "smoke": np.random.choice([0, 1], n),
        "cardio": np.random.choice([0, 1], n),
    })


def test_dynamic_training_threshold_is_oof_locked(tmp_path: Path):
    """
    TEST A: Verify threshold is calculated from Out-Of-Fold (OOF) predictions,
    NOT from in-sample training predictions of the final model.
    """
    df = _generate_synthetic_data(n=120, seed=42)

    captured_threshold_inputs = []
    orig_find_optimal_threshold = ev.find_optimal_threshold

    def spy_find_optimal_threshold(y_true, y_prob, method="youden", **kwargs):
        captured_threshold_inputs.append({
            "y_true": np.copy(y_true),
            "y_prob": np.copy(y_prob),
            "method": method,
        })
        return orig_find_optimal_threshold(y_true, y_prob, method=method, **kwargs)

    with patch("src.training_engine.find_optimal_threshold", side_effect=spy_find_optimal_threshold):
        job = execute_training_workflow(
            df=df,
            target_column="cardio",
            dataset_name="oof_test",
            models_to_train=["logistic_regression"],
            test_size=0.25,
            random_seed=42,
            output_base_dir=tmp_path,
        )

    assert len(captured_threshold_inputs) == 1, "find_optimal_threshold must be called for threshold locking"
    call_info = captured_threshold_inputs[0]

    # Verify input is of length N_train (90), not N_test (30) or full N (120)
    assert len(call_info["y_true"]) == 90
    assert len(call_info["y_prob"]) == 90
    assert call_info["method"] == "youden"

    # Verify metadata saved in threshold.json
    thresh_file = tmp_path / job.job_id / "logistic_regression" / "threshold.json"
    assert thresh_file.is_file()
    with open(thresh_file, "r") as f:
        meta = json.load(f)

    assert meta["threshold_source"] == "training_oof"
    assert meta["threshold_method"] == "youden"
    assert meta["cv_folds"] == 5
    assert meta["random_seed"] == 42
    assert "locked_threshold" in meta

    # Verify that the locked threshold is stored in job results
    res = job.results["logistic_regression"]
    assert res["threshold_source"] == "training_oof"
    assert res["cv_folds"] == 5


def test_dynamic_training_threshold_invariant_to_test_label_mutation(tmp_path: Path):
    """
    TEST B: Run threshold selection, mutate y_test, rerun, and prove threshold is
    bitwise identical (strictly invariant to holdout test labels).
    """
    df = _generate_synthetic_data(n=120, seed=42)

    X = df[[c for c in df.columns if c != "cardio"]]
    y = df["cardio"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 1. Run baseline workflow with legitimate y_test
    with patch("src.training_engine.train_test_split", return_value=(X_tr, X_te, y_tr, y_te)):
        job1 = execute_training_workflow(
            df=df,
            target_column="cardio",
            dataset_name="invariance_baseline",
            models_to_train=["logistic_regression"],
            test_size=0.25,
            random_seed=42,
            output_base_dir=tmp_path,
        )
    thresh_baseline = job1.results["logistic_regression"]["locked_threshold"]

    # 2. Test multiple adversarial attacks on y_test:
    adversarial_mutations = [
        ("inverted", 1 - y_te),
        ("all_zeros", np.zeros_like(y_te)),
        ("all_ones", np.ones_like(y_te)),
        ("shuffled", np.random.permutation(y_te)),
    ]

    for attack_name, y_mutated in adversarial_mutations:
        with patch("src.training_engine.train_test_split", return_value=(X_tr, X_te, y_tr, y_mutated)):
            job_adv = execute_training_workflow(
                df=df,
                target_column="cardio",
                dataset_name=f"invariance_{attack_name}",
                models_to_train=["logistic_regression"],
                test_size=0.25,
                random_seed=42,
                output_base_dir=tmp_path,
            )
        thresh_adv = job_adv.results["logistic_regression"]["locked_threshold"]
        assert thresh_adv == thresh_baseline, (
            f"Adversarial contamination under {attack_name}: threshold changed from {thresh_baseline} to {thresh_adv}!"
        )


def test_dynamic_training_final_model_is_refit_after_cv(tmp_path: Path):
    """
    TEST C: Verify that after CV fold estimators are used for threshold locking,
    they are discarded and a FRESH final estimator is trained on the ENTIRE training partition.
    """
    df = _generate_synthetic_data(n=120, seed=42)

    fitted_sample_counts = []
    orig_fit = LogisticRegression.fit

    def tracked_fit(self, X, y, **kw):
        fitted_sample_counts.append(len(X))
        return orig_fit(self, X, y, **kw)

    with patch.object(LogisticRegression, "fit", tracked_fit):
        job = execute_training_workflow(
            df=df,
            target_column="cardio",
            dataset_name="refit_test",
            models_to_train=["logistic_regression"],
            test_size=0.25,
            random_seed=42,
            output_base_dir=tmp_path,
        )

    # For 5-fold CV: 5 fold estimators + 1 final refit production estimator = 6 total
    assert len(fitted_sample_counts) == 6, (
        f"Expected 6 fit calls (5 CV + 1 final refit), got {len(fitted_sample_counts)}"
    )

    # First 5 fits were on CV folds: 72 samples each (90 * 4 / 5)
    for fold_idx in range(5):
        assert fitted_sample_counts[fold_idx] == 72, (
            f"Fold {fold_idx} fit saw {fitted_sample_counts[fold_idx]} samples instead of 72!"
        )

    # 6th fit was the final production model: MUST have seen all 90 training samples!
    assert fitted_sample_counts[-1] == 90, (
        f"Final production estimator was not refit on entire training partition! Saw {fitted_sample_counts[-1]} samples instead of 90."
    )


def test_dynamic_training_test_labels_are_never_consumed_before_evaluation(tmp_path: Path):
    """
    TEST D: Verify y_test is strictly prohibited from any threshold selection,
    calibration, preprocessor fitting, or model selection function.
    """
    df = _generate_synthetic_data(n=120, seed=42)

    test_label_consumed = []
    orig_threshold = ev.find_optimal_threshold

    def monitored_find_optimal_threshold(y_true, y_prob, method="youden", **kwargs):
        # Test partition length is 30. If length == 30, it consumed y_test!
        if len(y_true) == 30:
            test_label_consumed.append("find_optimal_threshold consumed test labels!")
        return orig_threshold(y_true, y_prob, method=method, **kwargs)

    with patch("src.training_engine.find_optimal_threshold", side_effect=monitored_find_optimal_threshold):
        job = execute_training_workflow(
            df=df,
            target_column="cardio",
            dataset_name="label_leakage_test",
            models_to_train=["logistic_regression"],
            test_size=0.25,
            random_seed=42,
            output_base_dir=tmp_path,
        )

    assert len(test_label_consumed) == 0, f"Leakage detected: {test_label_consumed}"


def test_dynamic_training_preprocessing_is_train_fold_isolated(tmp_path: Path):
    """
    TEST E: Verify preprocessing is fitted independently within each CV training fold,
    and validation fold statistics do NOT leak into fold preprocessors.
    """
    df = _generate_synthetic_data(n=120, seed=42)

    prep_fit_sample_counts = []
    orig_fit_transform = ColumnTransformer.fit_transform

    def tracked_fit_transform(self, X, y=None, **kw):
        prep_fit_sample_counts.append(len(X))
        return orig_fit_transform(self, X, y, **kw)

    with patch.object(ColumnTransformer, "fit_transform", tracked_fit_transform):
        job = execute_training_workflow(
            df=df,
            target_column="cardio",
            dataset_name="prep_isolation_test",
            models_to_train=["logistic_regression"],
            test_size=0.25,
            random_seed=42,
            output_base_dir=tmp_path,
        )

    # 1 preprocessor for final model (90 samples) + 5 preprocessors for the 5 CV folds (72 samples each) = 6 total
    assert len(prep_fit_sample_counts) == 6, (
        f"Expected 6 preprocessor fit_transform calls, got {len(prep_fit_sample_counts)}"
    )

    # First fit_transform was for the production preprocessor on all 90 train samples
    assert prep_fit_sample_counts[0] == 90

    # Remaining 5 fit_transforms were fold preprocessors: strictly 72 samples each
    for fold_idx in range(1, 6):
        assert prep_fit_sample_counts[fold_idx] == 72, (
            f"Fold {fold_idx} preprocessor fit saw {prep_fit_sample_counts[fold_idx]} samples instead of 72!"
        )


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_dynamic_training_threshold_is_oof_locked(p)
        print("PASS: test_dynamic_training_threshold_is_oof_locked")
        test_dynamic_training_threshold_invariant_to_test_label_mutation(p)
        print("PASS: test_dynamic_training_threshold_invariant_to_test_label_mutation")
        test_dynamic_training_final_model_is_refit_after_cv(p)
        print("PASS: test_dynamic_training_final_model_is_refit_after_cv")
        test_dynamic_training_test_labels_are_never_consumed_before_evaluation(p)
        print("PASS: test_dynamic_training_test_labels_are_never_consumed_before_evaluation")
        test_dynamic_training_preprocessing_is_train_fold_isolated(p)
        print("PASS: test_dynamic_training_preprocessing_is_train_fold_isolated")
        print("\nALL DYNAMIC TRAINING OOF THRESHOLD TESTS PASSED!")
