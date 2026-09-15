"""
Dynamic Dataset-to-Model Training Workflow Engine for CardioQ Platform.
Supports:
- User uploaded dataset training (Mode B) and canonical dataset training (Mode A)
- Explicit target selection and feature column selection
- Stratified train/test partitioning with reproducible random seed
- Prospective 5-fold Stratified Out-Of-Fold (OOF) threshold locking strictly on TRAIN
- Discarding fold estimators after threshold locking
- Fresh final production model refitted on entire training partition
- Unseen holdout test partition evaluation at locked prospective threshold
- Fold-isolated preprocessing (scaling, imputation, encoding)
- Classical model suite (Logistic Regression, Random Forest, HistGB, CatBoost)
- Quantum model suite (VQC, Hybrid QNN) with linear input projection
- Metric computation (ROC-AUC, PR-AUC, Accuracy, Sens, Spec, F1, Brier)
- Timing benchmarks (training time, inference latency per sample)
- Modular serialization under artifacts/training_jobs/<job_id>/
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from src.evaluate import evaluate_predictions, find_optimal_threshold
from src.models.hybrid_qnn import HybridQNNModel
from src.models.vqc_model import VQCModel
from config.settings import settings

TRAINING_JOBS_DIR = settings.TRAINING_JOBS_DIR


@dataclass
class ModelTrainingResult:
    model_key: str
    model_name: str
    model_type: str  # "Classical" or "Quantum"
    training_time_seconds: float
    inference_latency_ms: float
    locked_threshold: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    brier_score: float
    artifact_path: str
    threshold_source: str = "training_oof"
    threshold_method: str = "youden"
    cv_folds: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingJobSummary:
    job_id: str
    status: str  # "COMPLETED", "FAILED", "RUNNING"
    dataset_name: str
    dataset_records: int
    target_column: str
    feature_count: int
    feature_columns: List[str]
    train_samples: int
    test_samples: int
    random_seed: int
    models_trained: List[str]
    results: Dict[str, Dict[str, Any]]
    started_at: str
    completed_at: Optional[str]
    error_message: Optional[str] = None
    cv_folds: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Construct an unfitted, isolated preprocessing pipeline."""
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers)


def _create_estimator(
    m_key: str,
    random_seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[Any, str, str]:
    """Factory function for model estimators."""
    m_key_lower = m_key.lower()
    if m_key_lower == "logistic_regression":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=random_seed), "Logistic Regression", "Classical"
    elif m_key_lower == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_seed, n_jobs=-1), "Random Forest", "Classical"
    elif m_key_lower == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=random_seed), "HistGradientBoosting", "Classical"
    elif m_key_lower == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=100, depth=6, verbose=False, random_seed=random_seed), "CatBoost", "Classical"
    elif m_key_lower in ("vqc", "variational_quantum_classifier"):
        m_samples = max_samples if max_samples is not None else 500
        return VQCModel(n_qubits=4, n_layers=2, n_epochs=5, max_train_samples=m_samples, random_seed=random_seed), "Variational Quantum Classifier", "Quantum"
    elif m_key_lower in ("hybrid_qnn", "quantum_neural_network"):
        m_samples = max_samples if max_samples is not None else 500
        return HybridQNNModel(n_qubits=4, n_layers=2, n_epochs=5, max_train_samples=m_samples, random_seed=random_seed), "Hybrid Quantum Neural Network", "Quantum Hybrid"
    else:
        raise ValueError(f"Unsupported model family: '{m_key}'")


def _predict_probabilities(model: Any, X_proc: Any) -> np.ndarray:
    """Extract continuous positive class risk probabilities from an estimator."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_proc)[:, 1]
    elif hasattr(model, "predict_risk"):
        return model.predict_risk(X_proc)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_proc)
        return 1.0 / (1.0 + np.exp(-scores))
    else:
        return model.predict(X_proc).astype(float)


def execute_training_workflow(
    df: pd.DataFrame,
    target_column: str,
    dataset_name: str = "custom_dataset",
    selected_features: Optional[List[str]] = None,
    models_to_train: Optional[List[str]] = None,
    test_size: float = 0.20,
    random_seed: int = 42,
    enable_quantum: bool = False,
    output_base_dir: Optional[Path] = None,
    cv_folds: int = 5,
) -> TrainingJobSummary:
    """
    Execute full dataset-to-model training pipeline on uploaded/custom data.
    Methodological Standards:
    1. Stratified train/test split.
    2. 5-Fold Stratified Out-Of-Fold (OOF) cross-validation strictly on TRAIN.
    3. All preprocessing fitted strictly within each CV fold's training subset (zero leakage).
    4. Operational decision threshold locked strictly from y_train + OOF predictions via Youden J.
    5. Discard CV fold estimators.
    6. Fit a FRESH final production estimator on the ENTIRE training partition.
    7. Evaluate on untouched test partition at locked OOF threshold.
    8. y_test is NEVER used for threshold, model selection, calibration, or preprocessing.
    """
    start_time = time.time()
    started_at_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time))
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job_dir = (output_base_dir or TRAINING_JOBS_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if not target_column or not str(target_column).strip():
        raise ValueError("Target column must be explicitly specified and non-empty.")
    target_column = str(target_column).strip()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    # 1. Clean missing target rows (never fabricate target)
    clean_df = df.dropna(subset=[target_column]).copy()
    if clean_df.empty:
        raise ValueError("Dataset has no valid rows after dropping missing target records.")

    y_raw = clean_df[target_column]
    unique_targets = y_raw.unique()
    if len(unique_targets) != 2:
        raise ValueError(f"Target column '{target_column}' must have exactly 2 classes for binary classification, found: {unique_targets}")

    # Map target to 0 and 1 deterministically
    sorted_targets = sorted(unique_targets)
    target_mapping = {sorted_targets[0]: 0, sorted_targets[1]: 1}
    y = y_raw.map(target_mapping).to_numpy(dtype=np.int32)

    # 2. Select feature columns
    all_candidate_cols = [c for c in clean_df.columns if c != target_column]
    if selected_features:
        feat_cols = [c for c in selected_features if c in all_candidate_cols]
    else:
        feat_cols = all_candidate_cols

    if not feat_cols:
        raise ValueError("No feature columns available for training.")

    X = clean_df[feat_cols].copy()

    # 3. Stratified Train / Test Partitioning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feat_cols if c not in num_cols]

    # 4. Fit Fresh Production Preprocessor on ENTIRE Train Partition
    final_preprocessor = _build_preprocessor(num_cols, cat_cols)
    X_train_proc = final_preprocessor.fit_transform(X_train)
    X_test_proc = final_preprocessor.transform(X_test)

    # Save preprocessing artifact
    joblib.dump(final_preprocessor, job_dir / "preprocessor.joblib")

    # 5. Model Selection and Stratified 5-Fold OOF Threshold Locking on TRAIN ONLY
    default_models = ["logistic_regression", "random_forest", "gradient_boosting"]
    if enable_quantum:
        default_models.append("vqc")
    active_models = models_to_train or default_models

    min_class_samples = int(np.min(np.bincount(y_train)))
    effective_cv_folds = min(cv_folds, min_class_samples)
    if effective_cv_folds < 2:
        raise ValueError(
            f"Insufficient samples per class for stratified cross-validation (min class count: {min_class_samples})."
        )

    skf = StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=random_seed)

    results: Dict[str, Dict[str, Any]] = {}

    for m_key in active_models:
        t0 = time.time()
        m_key_lower = m_key.lower()

        # Step A: 5-Fold Stratified Out-Of-Fold (OOF) Prediction on TRAIN ONLY
        oof_probs = np.zeros(len(y_train), dtype=np.float64)

        for fold_idx, (tr_fold_idx, val_fold_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr_fold = X_train.iloc[tr_fold_idx]
            y_tr_fold = y_train[tr_fold_idx]
            X_val_fold = X_train.iloc[val_fold_idx]

            # Fold-isolated preprocessing fitted STRICTLY on fold training subset
            fold_prep = _build_preprocessor(num_cols, cat_cols)
            X_tr_proc = fold_prep.fit_transform(X_tr_fold)
            X_val_proc = fold_prep.transform(X_val_fold)

            # Fit fold estimator
            fold_model, _, _ = _create_estimator(
                m_key_lower,
                random_seed=random_seed,
                max_samples=min(500, len(X_tr_fold)),
            )
            fold_model.fit(X_tr_proc, y_tr_fold)

            # Predict on unseen validation fold
            oof_probs[val_fold_idx] = _predict_probabilities(fold_model, X_val_proc)

        # Step B: Lock Operational Decision Threshold STRICTLY from OOF Predictions
        opt_thresh_dict = find_optimal_threshold(y_train, oof_probs, method="youden")
        tau_star = float(opt_thresh_dict["threshold"])
        j_val = float(opt_thresh_dict["youden_j"])

        # Step C: Fit a FRESH Final Production Model on the ENTIRE Training Partition
        final_model, model_name, model_type = _create_estimator(
            m_key_lower,
            random_seed=random_seed,
            max_samples=min(500, len(X_train)),
        )
        final_model.fit(X_train_proc, y_train)

        train_dur = time.time() - t0

        # Step D: Evaluate Final Model on Untouched Test Partition at Locked OOF Threshold
        t_inf_0 = time.time()
        probs_test = _predict_probabilities(final_model, X_test_proc)
        inf_latency_ms = ((time.time() - t_inf_0) / max(len(X_test), 1)) * 1000.0

        y_pred_test = (probs_test >= tau_star).astype(int)
        eval_metrics = evaluate_predictions(y_test, y_pred_test, probs_test)

        # Step E: Save Model and Explicit Threshold Metadata
        m_dir = job_dir / m_key_lower
        m_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, m_dir / "model.joblib")
        with open(m_dir / "threshold.json", "w") as f:
            json.dump({
                "locked_threshold": tau_star,
                "threshold_method": "youden",
                "threshold_source": "training_oof",
                "cv_folds": effective_cv_folds,
                "random_seed": random_seed,
                "youden_j": j_val,
            }, f, indent=2)
        with open(m_dir / "metadata.json", "w") as f:
            json.dump({
                "model_key": m_key_lower,
                "model_name": model_name,
                "target_column": target_column,
                "dataset_name": dataset_name,
                "locked_threshold": tau_star,
                "cv_folds": effective_cv_folds,
                "random_seed": random_seed,
            }, f, indent=2)

        res = ModelTrainingResult(
            model_key=m_key_lower,
            model_name=model_name,
            model_type=model_type,
            training_time_seconds=round(train_dur, 3),
            inference_latency_ms=round(inf_latency_ms, 3),
            locked_threshold=round(tau_star, 4),
            roc_auc=round(eval_metrics.roc_auc, 4),
            pr_auc=round(eval_metrics.pr_auc, 4),
            accuracy=round(eval_metrics.accuracy, 4),
            sensitivity=round(eval_metrics.sensitivity, 4),
            specificity=round(eval_metrics.specificity, 4),
            precision=round(eval_metrics.precision, 4),
            f1=round(eval_metrics.f1, 4),
            brier_score=round(eval_metrics.brier_score, 4),
            artifact_path=str(m_dir),
            threshold_source="training_oof",
            threshold_method="youden",
            cv_folds=effective_cv_folds,
        )
        results[m_key_lower] = res.to_dict()

    completed_at_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))

    summary = TrainingJobSummary(
        job_id=job_id,
        status="COMPLETED",
        dataset_name=dataset_name,
        dataset_records=len(clean_df),
        target_column=target_column,
        feature_count=len(feat_cols),
        feature_columns=feat_cols,
        train_samples=len(X_train),
        test_samples=len(X_test),
        random_seed=random_seed,
        models_trained=list(results.keys()),
        results=results,
        started_at=started_at_str,
        completed_at=completed_at_str,
        cv_folds=effective_cv_folds,
    )

    with open(job_dir / "job.json", "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    with open(job_dir / "job_summary.json", "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    training_config = {
        "job_id": job_id,
        "dataset_name": dataset_name,
        "target_column": target_column,
        "feature_columns": feat_cols,
        "test_size": test_size,
        "random_seed": random_seed,
        "cv_folds": effective_cv_folds,
        "enable_quantum": enable_quantum,
        "models_to_train": models_to_train,
        "started_at": started_at_str,
        "completed_at": completed_at_str,
    }
    with open(job_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    return summary
