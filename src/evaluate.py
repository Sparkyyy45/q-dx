"""
Clinical evaluation metrics, calibration assessment, and Out-Of-Fold (OOF) prediction store.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for cardiovascular risk classification."""
    roc_auc: float
    pr_auc: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float  # Recall / TPR
    specificity: float  # TNR
    precision: float  # PPV
    npv: float  # Negative Predictive Value
    f1: float
    brier_score: float
    expected_calibration_error: float
    confusion_matrix: List[List[int]]
    tn: int
    fp: int
    fn: int
    tp: int
    applied_threshold: float = 0.50
    bootstrap_ci: Optional[Dict[str, Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    ECE = sum_{b=1}^B (N_b / N) * |acc(b) - conf(b)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bins) - 1
    # Clip any exact 1.0 predictions into highest bin
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_assignments == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> EvaluationMetrics:
    """
    Compute full suite of medical classification and probabilistic calibration metrics.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    y_pr = np.asarray(y_prob, dtype=np.float64)

    # ROC-AUC and PR-AUC
    has_both_classes = len(np.unique(y_t)) > 1
    roc_auc = float(roc_auc_score(y_t, y_pr)) if has_both_classes else 0.0
    pr_auc = float(average_precision_score(y_t, y_pr)) if has_both_classes else 0.0

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    recall = float(recall_score(y_t, y_p, zero_division=0))
    precision = float(precision_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    brier = float(brier_score_loss(y_t, y_pr))
    ece = calculate_expected_calibration_error(y_t, y_pr, n_bins=n_bins)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return EvaluationMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        sensitivity=recall,
        specificity=specificity,
        precision=precision,
        npv=npv,
        f1=f1,
        brier_score=brier,
        expected_calibration_error=ece,
        confusion_matrix=[[tn, fp], [fn, tp]],
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def compute_bootstrap_ci(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.50,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute non-parametric bootstrap 95% confidence intervals for clinical metrics.
    Guarantees prospective, non-peeking uncertainty quantification on holdout test partitions.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_pr = np.asarray(y_prob, dtype=np.float64)
    n = len(y_t)
    rng = np.random.default_rng(seed)

    metric_samples: Dict[str, List[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "f1": [],
        "brier_score": [],
        "expected_calibration_error": [],
    }

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        y_b = y_t[boot_idx]
        pr_b = y_pr[boot_idx]

        if len(np.unique(y_b)) < 2:
            continue

        pred_b = (pr_b >= threshold).astype(np.int64)

        roc = float(roc_auc_score(y_b, pr_b))
        pr = float(average_precision_score(y_b, pr_b))

        tp = int(np.sum((y_b == 1) & (pred_b == 1)))
        tn = int(np.sum((y_b == 0) & (pred_b == 0)))
        fp = int(np.sum((y_b == 0) & (pred_b == 1)))
        fn = int(np.sum((y_b == 1) & (pred_b == 0)))

        acc = float((tp + tn) / n)
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        brier = float(np.mean((pr_b - y_b) ** 2))
        ece = calculate_expected_calibration_error(y_b, pr_b)

        metric_samples["roc_auc"].append(roc)
        metric_samples["pr_auc"].append(pr)
        metric_samples["accuracy"].append(acc)
        metric_samples["sensitivity"].append(sens)
        metric_samples["specificity"].append(spec)
        metric_samples["f1"].append(f1)
        metric_samples["brier_score"].append(brier)
        metric_samples["expected_calibration_error"].append(ece)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_dict: Dict[str, Dict[str, float]] = {}
    for m_name, vals in metric_samples.items():
        if len(vals) > 0:
            arr = np.array(vals)
            ci_dict[m_name] = {
                "mean": float(np.mean(arr)),
                "ci_lower": float(np.percentile(arr, lower_pct)),
                "ci_upper": float(np.percentile(arr, upper_pct)),
            }
        else:
            ci_dict[m_name] = {
                "mean": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }

    return ci_dict


def evaluate_at_locked_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    locked_threshold: float = 0.50,
    n_bins: int = 10,
    compute_ci: bool = False,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> EvaluationMetrics:
    """
    Evaluate predictions strictly at a pre-specified locked threshold (preventing test-set threshold peeking).
    Optionally computes non-parametric bootstrap confidence intervals.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_pr = np.asarray(y_prob, dtype=np.float64)
    y_p = (y_pr >= locked_threshold).astype(np.int64)

    metrics = evaluate_predictions(y_t, y_p, y_pr, n_bins=n_bins)
    metrics.applied_threshold = float(locked_threshold)

    if compute_ci:
        metrics.bootstrap_ci = compute_bootstrap_ci(
            y_t, y_pr, threshold=locked_threshold, n_bootstraps=n_bootstraps, seed=seed
        )

    return metrics


class OutOfFoldStore:
    """
    Central repository for storing and evaluating out-of-fold (OOF) cross-validation predictions.
    Guarantees that all models are evaluated on identical patient folds.
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.y_true = np.zeros(n_samples, dtype=np.int64)
        self.is_label_set = False

        # Model name -> (oof_pred array, oof_prob array, fold_metrics_list)
        self.oof_predictions: Dict[str, np.ndarray] = {}
        self.oof_probabilities: Dict[str, np.ndarray] = {}
        self.fold_metrics: Dict[str, List[EvaluationMetrics]] = {}

    def register_labels(self, indices: np.ndarray, y: np.ndarray | pd.Series) -> None:
        """Register ground truth labels for the fold."""
        self.y_true[indices] = np.asarray(y, dtype=np.int64)
        self.is_label_set = True

    def record_fold(
        self,
        model_name: str,
        val_indices: np.ndarray,
        y_val: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        """Record fold predictions and probabilities for a specific model."""
        if model_name not in self.oof_predictions:
            self.oof_predictions[model_name] = np.zeros(self.n_samples, dtype=np.int64)
            self.oof_probabilities[model_name] = np.zeros(self.n_samples, dtype=np.float64)
            self.fold_metrics[model_name] = []

        self.oof_predictions[model_name][val_indices] = np.asarray(y_pred, dtype=np.int64)
        self.oof_probabilities[model_name][val_indices] = np.asarray(y_prob, dtype=np.float64)

        if not self.is_label_set:
            self.y_true[val_indices] = np.asarray(y_val, dtype=np.int64)

        fold_metric = evaluate_predictions(y_val, y_pred, y_prob)
        self.fold_metrics[model_name].append(fold_metric)

    def get_oof_metrics(self, model_name: str) -> EvaluationMetrics:
        """Calculate overall Out-Of-Fold evaluation metrics for a model."""
        if model_name not in self.oof_predictions:
            raise KeyError(f"No records found for model: {model_name}")
        return evaluate_predictions(
            self.y_true,
            self.oof_predictions[model_name],
            self.oof_probabilities[model_name],
        )

    def get_fold_statistics(self, model_name: str) -> Dict[str, Tuple[float, float]]:
        """Calculate mean and std of each metric across CV folds."""
        metrics_list = self.fold_metrics.get(model_name, [])
        if not metrics_list:
            return {}

        keys = [
            "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
            "sensitivity", "specificity", "precision", "f1", "brier_score", "expected_calibration_error"
        ]
        stats = {}
        for k in keys:
            vals = [getattr(m, k) for m in metrics_list]
            stats[k] = (float(np.mean(vals)), float(np.std(vals)))
        return stats

    def create_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comprehensive comparison DataFrame across all models
        based on pooled Out-Of-Fold predictions.
        """
        rows = []
        for name in self.oof_predictions.keys():
            oof_m = self.get_oof_metrics(name)
            fold_stats = self.get_fold_statistics(name)
            rows.append({
                "Model": name,
                "ROC-AUC": f"{oof_m.roc_auc:.4f} (±{fold_stats.get('roc_auc', (0,0))[1]:.3f})",
                "PR-AUC": f"{oof_m.pr_auc:.4f} (±{fold_stats.get('pr_auc', (0,0))[1]:.3f})",
                "Brier Score": f"{oof_m.brier_score:.4f}",
                "ECE": f"{oof_m.expected_calibration_error:.4f}",
                "Sensitivity (Recall)": f"{oof_m.sensitivity:.4f}",
                "Specificity": f"{oof_m.specificity:.4f}",
                "Precision": f"{oof_m.precision:.4f}",
                "F1-Score": f"{oof_m.f1:.4f}",
                "Accuracy": f"{oof_m.accuracy:.4f}",
                "Balanced Accuracy": f"{oof_m.balanced_accuracy:.4f}",
                "_raw_auc": oof_m.roc_auc,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="_raw_auc", ascending=False).drop(columns=["_raw_auc"])
        return df

    def get_optimal_threshold(
        self,
        model_name: str,
        method: str = "youden",
        fn_cost: float = 1.0,
        fp_cost: float = 1.0,
    ) -> Dict[str, float]:
        """
        Derive prospective optimal threshold strictly from out-of-fold training predictions.
        Prevents test-set label contamination by freezing the threshold before holdout evaluation.
        """
        if model_name not in self.oof_probabilities:
            raise KeyError(f"No OOF predictions registered for model '{model_name}'")
        return find_optimal_threshold(
            self.y_true,
            self.oof_probabilities[model_name],
            method=method,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )

    def select_champion_model(
        self,
        primary_metric: str = "roc_auc",
    ) -> Tuple[str, EvaluationMetrics]:
        """
        Select the production champion model strictly from development out-of-fold (OOF) cross-validation.
        Predefined Selection Protocol:
          1. Highest OOF ROC-AUC
          2. Tie-breaker 1: Highest OOF PR-AUC
          3. Tie-breaker 2: Lowest OOF Brier Score

        Guarantees that holdout test labels are never accessed during model selection.
        """
        if not self.oof_probabilities:
            raise ValueError("No models registered in OutOfFoldStore.")

        best_name: Optional[str] = None
        best_metrics: Optional[EvaluationMetrics] = None
        best_rank = (-1.0, -1.0, float("inf"))

        for name in self.oof_probabilities.keys():
            m = self.get_oof_metrics(name)
            primary_val = getattr(m, primary_metric, 0.0)
            rank = (float(primary_val), float(m.pr_auc), -float(m.brier_score))
            if rank > best_rank:
                best_rank = rank
                best_name = name
                best_metrics = m

        if best_name is None or best_metrics is None:
            raise RuntimeError("Failed to select champion model from OOF store.")

        return best_name, best_metrics


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    method: str = "youden",
    fn_cost: float = 1.0,
    fp_cost: float = 1.0,
) -> Dict[str, float]:
    """
    Compute the classification threshold from an ROC curve that
    optimizes a chosen criterion.

    method="youden": maximize Youden's J = sensitivity + specificity - 1
    method="cost_weighted": maximize (sensitivity * fn_cost) -
        (1 - specificity) * fp_cost, allowing asymmetric costs for
        false negatives vs false positives (e.g. missing a CVD case
        may be costlier than a false alarm) — default fn_cost=fp_cost=1.0
        reduces to a variant of Youden's J.

    Returns: dict with keys {threshold, sensitivity, specificity,
    youden_j (or cost_weighted_score)} at the optimal point.
    Must use sklearn.metrics.roc_curve internally, not a manual
    threshold sweep, for numerical consistency with the AUC already
    being reported.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_pr = np.asarray(y_prob, dtype=np.float64)

    if len(np.unique(y_t)) < 2:
        return {
            "threshold": 0.5,
            "sensitivity": 0.0,
            "specificity": 1.0,
            "youden_j": 0.0,
            "cost_weighted_score": 0.0,
        }

    fpr, tpr, thresholds = roc_curve(y_t, y_pr)

    if method == "youden":
        scores = tpr - fpr
    elif method == "cost_weighted":
        # (sensitivity * fn_cost) - (1 - specificity) * fp_cost
        # Since (1 - specificity) == fpr:
        scores = (tpr * fn_cost) - (fpr * fp_cost)
    else:
        raise ValueError(f"Unsupported method: '{method}'. Choose 'youden' or 'cost_weighted'.")

    best_idx = int(np.argmax(scores))
    opt_thresh = float(thresholds[best_idx])
    if np.isinf(opt_thresh) or opt_thresh > 1.0:
        opt_thresh = 1.0
    elif opt_thresh < 0.0:
        opt_thresh = 0.0

    sensitivity = float(tpr[best_idx])
    specificity = float(1.0 - fpr[best_idx])
    youden_j = float(tpr[best_idx] - fpr[best_idx])
    cost_weighted_score = float((tpr[best_idx] * fn_cost) - (fpr[best_idx] * fp_cost))

    return {
        "threshold": opt_thresh,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_j": youden_j,
        "cost_weighted_score": cost_weighted_score,
    }


def create_threshold_recalibration_table(
    internal_metrics: Dict[str, EvaluationMetrics],
    internal_thresholds: Dict[str, Dict[str, float]],
    external_metrics: Optional[Dict[str, EvaluationMetrics]] = None,
    external_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    internal_prevalence: Optional[float] = None,
    external_prevalence: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate a side-by-side comparison table contrasting fixed (0.50) vs optimal recalibrated thresholds
    for both internal holdout and external Framingham cohorts.
    """
    rows = []
    int_prev_str = f" ({internal_prevalence:.1%})" if internal_prevalence is not None else ""
    ext_prev_str = f" ({external_prevalence:.1%})" if external_prevalence is not None else ""

    for model_name, int_m in internal_metrics.items():
        int_opt = internal_thresholds.get(model_name, {})
        rows.append({
            "Model": model_name,
            "Cohort": f"Internal Holdout{int_prev_str}",
            "Fixed Cutoff": "0.5000",
            "Fixed Sens": f"{int_m.sensitivity:.4f}",
            "Fixed Spec": f"{int_m.specificity:.4f}",
            "Optimal Cutoff": f"{int_opt.get('threshold', float('nan')):.4f}",
            "Optimal Sens": f"{int_opt.get('sensitivity', float('nan')):.4f}",
            "Optimal Spec": f"{int_opt.get('specificity', float('nan')):.4f}",
            "Youden's J": f"{int_opt.get('youden_j', float('nan')):.4f}",
        })

        if external_metrics and external_thresholds and model_name in external_metrics:
            ext_m = external_metrics[model_name]
            ext_opt = external_thresholds.get(model_name, {})
            rows.append({
                "Model": model_name,
                "Cohort": f"External Framingham{ext_prev_str}",
                "Fixed Cutoff": "0.5000",
                "Fixed Sens": f"{ext_m.sensitivity:.4f}",
                "Fixed Spec": f"{ext_m.specificity:.4f}",
                "Optimal Cutoff": f"{ext_opt.get('threshold', float('nan')):.4f}",
                "Optimal Sens": f"{ext_opt.get('sensitivity', float('nan')):.4f}",
                "Optimal Spec": f"{ext_opt.get('specificity', float('nan')):.4f}",
                "Youden's J": f"{ext_opt.get('youden_j', float('nan')):.4f}",
            })

    return pd.DataFrame(rows)


__all__ = [
    "EvaluationMetrics",
    "OutOfFoldStore",
    "calculate_expected_calibration_error",
    "evaluate_predictions",
    "compute_bootstrap_ci",
    "evaluate_at_locked_threshold",
    "find_optimal_threshold",
    "create_threshold_recalibration_table",
]

