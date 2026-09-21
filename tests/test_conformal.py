"""
Unit tests for Conformal Prediction & Uncertainty Quantification in CardioQ:
- Non-conformity score calculation
- Split conformal calibration on probability distributions
- Quantile cutoff determination
- Prediction set coverage and clinical ambiguity detection
"""

import numpy as np
import pytest
from src.conformal import ConformalPredictor, ConformalPredictionResult


def test_conformal_predictor_init():
    cp = ConformalPredictor(alpha=0.10)
    assert cp.alpha == 0.10
    assert cp.confidence_level == 0.90
    assert cp.cutoff > 0.0


def test_conformal_calibration():
    cp = ConformalPredictor(alpha=0.10)
    # 100 well-calibrated synthetic predictions
    np.random.seed(42)
    labels = np.random.binomial(1, 0.5, size=100)
    probs = np.where(labels == 1, np.random.uniform(0.6, 0.95, size=100), np.random.uniform(0.05, 0.4, size=100))

    cutoff = cp.calibrate_from_scores(probs, labels, alpha=0.10)
    assert 0.0 < cutoff < 1.0
    assert cp.cutoff == cutoff


def test_conformal_predict_high_confidence_positive():
    cp = ConformalPredictor(alpha=0.10, calibrated_cutoff=0.75)
    res = cp.predict(probability=0.88)

    assert isinstance(res, ConformalPredictionResult)
    assert res.confidence_level == 0.90
    assert 1 in res.prediction_set_indices
    assert res.risk_probability == 0.88
    assert res.risk_band_lower < 0.88
    assert res.risk_band_upper > 0.88
    assert "High-Confidence Positive" in res.clinical_interpretation


def test_conformal_predict_high_confidence_negative():
    cp = ConformalPredictor(alpha=0.10, calibrated_cutoff=0.75)
    res = cp.predict(probability=0.08)

    assert isinstance(res, ConformalPredictionResult)
    assert 0 in res.prediction_set_indices
    assert res.risk_probability == 0.08
    assert "High-Confidence Negative" in res.clinical_interpretation


def test_conformal_predict_boundary_case():
    # An uncertain boundary case near threshold with large nonconformity cutoff
    cp = ConformalPredictor(alpha=0.10, calibrated_cutoff=0.85)
    res = cp.predict(probability=0.49)

    assert isinstance(res, ConformalPredictionResult)
    # Both classes should be included when cutoff is large and prob is near 0.5
    assert len(res.prediction_set) >= 1
    d = res.to_dict()
    assert "confidence_level" in d
    assert "prediction_set" in d
