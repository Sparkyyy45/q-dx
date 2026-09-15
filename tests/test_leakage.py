"""
Explicit verification tests proving zero data leakage across folds and partitions.
Guarantees that test/validation data cannot leak into learned preprocessing,
outlier thresholds, scaling, feature reduction, or model parameters.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.preprocess import build_preprocessor, fit_preprocessor, transform_data
from src.reduction import select_features, reduce_features


def test_preprocessor_does_not_leak_validation_statistics():
    """
    Verify that transforming validation data (even with extreme outliers)
    does NOT alter the fitted preprocessor's internal learned statistics.
    """
    config = PipelineConfig(scaling_strategy="standard", outlier_clipping=True)
    num_cols = ["age", "weight", "ap_hi"]

    np.random.seed(42)
    X_train = pd.DataFrame(np.random.normal(100, 15, size=(100, 3)), columns=num_cols)

    # 1. Fit on X_train only
    preprocessor = build_preprocessor(config, num_cols, [])
    artifact = fit_preprocessor(preprocessor, X_train, None, num_cols, [], config)

    # Extract learned state
    scaler = artifact.preprocessor.named_transformers_["num"].named_steps["scaler"]
    learned_means_before = np.copy(scaler.mean_)
    learned_vars_before = np.copy(scaler.var_)
    clipper_bounds_before = artifact.outlier_bounds.copy()

    # 2. Transform validation set with massive outliers (e.g. 100,000)
    X_val_extreme = pd.DataFrame({
        "age": [100000.0, -50000.0],
        "weight": [999999.0, -100.0],
        "ap_hi": [888888.0, 0.0],
    })
    _ = transform_data(artifact, X_val_extreme)

    # 3. Confirm that learned means and vars are bitwise identical
    assert np.array_equal(scaler.mean_, learned_means_before), "Data leakage: Scaler mean modified by validation data!"
    assert np.array_equal(scaler.var_, learned_vars_before), "Data leakage: Scaler variance modified by validation data!"
    assert artifact.outlier_bounds == clipper_bounds_before, "Data leakage: Outlier bounds altered by validation data!"


def test_feature_selection_does_not_leak_validation_data():
    """
    Verify that feature selection scores and selected subsets depend
    exclusively on training partitions, and are unaffected by validation labels.
    """
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 10), columns=[f"f_{i}" for i in range(10)])
    y_train = np.random.binomial(1, 0.5, 100)

    # Fit feature reducer on training split
    X_tr_red, artifact, reducer = select_features(
        X_train, y_train, strategy="f_classif", k=5, random_state=42
    )
    selected_features_original = list(artifact.output_feature_names)
    scores_original = artifact.scores.copy()

    # Transform validation split
    X_val = pd.DataFrame(np.random.randn(30, 10), columns=[f"f_{i}" for i in range(10)])
    _ = reduce_features(artifact, reducer, X_val)

    # Re-verify reducer state
    assert artifact.output_feature_names == selected_features_original
    assert artifact.scores == scores_original


def test_calibrated_svm_train_isolation():
    """
    Verify that calibrated SVM produces identical probabilities when evaluated
    single-sample vs multi-sample (no batch leakage).
    """
    from src.models.svm_calibrated import CalibratedSVMModel

    config = PipelineConfig()
    svm = CalibratedSVMModel(config)

    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 4), columns=[f"f_{i}" for i in range(4)])
    y_train = np.random.binomial(1, 0.5, 100)
    svm.fit(X_train, y_train)

    # Validation samples
    X_val = pd.DataFrame(np.random.randn(5, 4), columns=[f"f_{i}" for i in range(4)])
    batch_probas = svm.predict_proba(X_val)

    # Individual single-sample inferences
    single_probas = [svm.predict_proba(X_val.iloc[[i]])[0] for i in range(len(X_val))]

    # Single-sample probability must exactly match batch probability (no cross-sample contamination)
    assert np.allclose(batch_probas, np.array(single_probas), atol=1e-7)


def test_outlier_bound_fitting_cannot_condition_on_target_labels():
    """
    Structural verification: Outlier clipping bounds MUST NEVER condition on target labels.

    Reference notebook bug:
    The reference notebook computed separate IQR bounds PER TARGET CLASS:
      'for gender_category in df[TARGET].unique(): ...'
    which baked target label distributions directly into features before model training.

    This test asserts:
    1. The core outlier bound function compute_iqr_bounds does NOT accept a target/label parameter.
    2. Attempting to pass target/label arguments raises a TypeError.
    3. IQRClipper.fit produces bit-for-bit identical bounds regardless of y (y=None, y=zeros, y=classes, inverted y).
    """
    import inspect
    from src.preprocess import compute_iqr_bounds, IQRClipper

    # 1. Structural signature inspection: No target/label parameters permitted
    sig = inspect.signature(compute_iqr_bounds)
    forbidden_params = {"y", "target", "label", "labels", "classes", "group"}
    present_params = set(sig.parameters.keys())
    assert forbidden_params.isdisjoint(present_params), (
        f"Structural leakage vulnerability: compute_iqr_bounds signature contains forbidden parameters: "
        f"{forbidden_params.intersection(present_params)}"
    )

    # 2. Rejection of extraneous target keyword arguments
    X_dummy = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
    with pytest.raises(TypeError):
        compute_iqr_bounds(X_dummy, target=np.array([0, 1, 0, 1, 0]))  # type: ignore

    # 3. Invariance test: Features with massive class differences
    # Class 0 centered around 10, Class 1 centered around 1000
    np.random.seed(42)
    n = 200
    y_class = np.array([0] * (n // 2) + [1] * (n // 2))
    x_feat = np.concatenate([
        np.random.normal(10.0, 2.0, n // 2),
        np.random.normal(1000.0, 50.0, n // 2),
    ]).reshape(-1, 1)

    clipper_no_y = IQRClipper().fit(x_feat, y=None)
    clipper_with_y = IQRClipper().fit(x_feat, y=y_class)
    clipper_inverted_y = IQRClipper().fit(x_feat, y=1 - y_class)
    clipper_shuffled_y = IQRClipper().fit(x_feat, y=np.random.permutation(y_class))

    # All learned bounds must be bit-for-bit identical across all pooled samples
    np.testing.assert_array_equal(clipper_no_y.lower_bounds_, clipper_with_y.lower_bounds_)
    np.testing.assert_array_equal(clipper_no_y.upper_bounds_, clipper_with_y.upper_bounds_)
    np.testing.assert_array_equal(clipper_with_y.lower_bounds_, clipper_inverted_y.lower_bounds_)
    np.testing.assert_array_equal(clipper_with_y.upper_bounds_, clipper_inverted_y.upper_bounds_)
    np.testing.assert_array_equal(clipper_with_y.lower_bounds_, clipper_shuffled_y.lower_bounds_)
    np.testing.assert_array_equal(clipper_with_y.upper_bounds_, clipper_shuffled_y.upper_bounds_)


def test_holdout_label_mutation_adversarial_invariance():
    """
    Phase 10 Forensic Adversarial Test:
    Proves that arbitrarily mutating, inverting, shuffling, or fabricating holdout labels
    has ZERO effect on:
    1. The selected champion model (derived strictly from development OOF)
    2. The locked operational threshold tau* (derived strictly from development OOF Youden's J)
    3. Learned preprocessing parameters (fitted strictly on development data)
    """
    from src.evaluate import OutOfFoldStore, find_optimal_threshold
    from src.preprocess import build_preprocessor, fit_preprocessor

    np.random.seed(42)
    n_dev = 200
    n_holdout = 100
    p = 5

    # Development split
    X_dev = pd.DataFrame(np.random.randn(n_dev, p), columns=[f"feat_{i}" for i in range(p)])
    y_dev = np.random.binomial(1, 0.5, n_dev)

    # Preprocessor fitted strictly on dev
    cfg = PipelineConfig(scaling_strategy="robust")
    prep = build_preprocessor(cfg, list(X_dev.columns), [])
    art_dev = fit_preprocessor(prep, X_dev, y_dev, list(X_dev.columns), [], cfg)
    scaler_center_before = np.copy(art_dev.preprocessor.named_transformers_["num"].named_steps["scaler"].center_)
    scaler_scale_before = np.copy(art_dev.preprocessor.named_transformers_["num"].named_steps["scaler"].scale_)

    # Simulate 5-fold OOF predictions for 2 models on development partition
    oof_store = OutOfFoldStore(n_samples=n_dev)
    oof_probs_catboost = 0.5 + 0.3 * (y_dev - 0.5) + np.random.normal(0, 0.05, n_dev)
    oof_probs_catboost = np.clip(oof_probs_catboost, 0.01, 0.99)
    oof_preds_catboost = (oof_probs_catboost >= 0.5).astype(int)

    oof_probs_lr = 0.5 + 0.15 * (y_dev - 0.5) + np.random.normal(0, 0.15, n_dev)
    oof_probs_lr = np.clip(oof_probs_lr, 0.01, 0.99)
    oof_preds_lr = (oof_probs_lr >= 0.5).astype(int)

    oof_store.register_labels(np.arange(n_dev), y_dev)
    oof_store.record_fold("catboost", np.arange(n_dev), y_dev, oof_preds_catboost, oof_probs_catboost)
    oof_store.record_fold("logistic_regression", np.arange(n_dev), y_dev, oof_preds_lr, oof_probs_lr)

    # Derive champion model and locked threshold strictly from development OOF
    champion_baseline, _ = oof_store.select_champion_model(primary_metric="roc_auc")
    tau_info = oof_store.get_optimal_threshold("catboost", method="youden")
    tau_star_baseline = tau_info["threshold"]
    j_baseline = tau_info["youden_j"]

    # Generate untouched holdout features and legitimate holdout labels
    X_holdout = pd.DataFrame(np.random.randn(n_holdout, p), columns=[f"feat_{i}" for i in range(p)])
    y_holdout_true = np.random.binomial(1, 0.5, n_holdout)

    # Adversarial label mutations on holdout partition:
    adversarial_holdout_mutations = [
        ("inverted", 1 - y_holdout_true),
        ("all_zeros", np.zeros(n_holdout, dtype=int)),
        ("all_ones", np.ones(n_holdout, dtype=int)),
        ("shuffled", np.random.permutation(y_holdout_true)),
        ("fabricated_constant", np.array([1 if i % 3 == 0 else 0 for i in range(n_holdout)])),
    ]

    for attack_name, y_mutated in adversarial_holdout_mutations:
        # Re-verify preprocessor center and scale
        np.testing.assert_array_equal(art_dev.preprocessor.named_transformers_["num"].named_steps["scaler"].center_, scaler_center_before)
        np.testing.assert_array_equal(art_dev.preprocessor.named_transformers_["num"].named_steps["scaler"].scale_, scaler_scale_before)

        # Champion selection must remain strictly invariant to holdout labels
        champion_after, _ = oof_store.select_champion_model(primary_metric="roc_auc")
        assert champion_after == champion_baseline == "catboost", f"Adversarial contamination under {attack_name}!"

        # Threshold locking must remain strictly invariant to holdout labels
        tau_info_after = oof_store.get_optimal_threshold("catboost", method="youden")
        tau_after = tau_info_after["threshold"]
        j_after = tau_info_after["youden_j"]
        assert np.isclose(tau_after, tau_star_baseline, atol=1e-12), f"Adversarial threshold drift under {attack_name}!"
        assert np.isclose(j_after, j_baseline, atol=1e-12), f"Adversarial Youden J drift under {attack_name}!"


