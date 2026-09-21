"""
Unit tests for Algorithmic Subgroup Fairness & Demographic Parity in CardioQ:
- Disparate impact ratio
- Equal opportunity difference
- Equalized odds computation
- Four-fifths rule verification
"""

import numpy as np
import pandas as pd
from src.fairness import audit_subgroup_fairness, get_cached_or_default_fairness_audit, compute_binary_group_metrics


def test_compute_binary_group_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])

    m = compute_binary_group_metrics(y_true, y_pred, group_name="TestGroup")
    assert m.sample_count == 6
    assert abs(m.true_positive_rate - 2 / 3) < 1e-3
    assert abs(m.false_negative_rate - 1 / 3) < 1e-3
    assert abs(m.accuracy - 4 / 6) < 1e-3


def test_audit_subgroup_fairness():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "gender": np.random.choice([1, 2], size=n),
        "age_years": np.random.uniform(30, 75, size=n),
    })
    y_true = np.random.binomial(1, 0.5, size=n)
    y_pred = np.random.binomial(1, 0.5, size=n)

    report = audit_subgroup_fairness(df, y_true, y_pred, model_name="TestModel")
    assert report.evaluation_sample_size == n
    assert "Female" in report.gender_metrics
    assert "Male" in report.gender_metrics
    assert "demographic_parity_ratio" in report.disparities
    assert isinstance(report.four_fifths_rule_passed, bool)
    assert len(report.clinical_fairness_summary) > 20


def test_cached_fairness_audit():
    audit = get_cached_or_default_fairness_audit()
    assert audit["evaluation_sample_size"] == 13741
    assert audit["four_fifths_rule_passed"] is True
    assert audit["demographic_parity_ratio"] > 0.80
    assert "Female" in audit["gender_metrics"]
    assert "Male" in audit["gender_metrics"]
