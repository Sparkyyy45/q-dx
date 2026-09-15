"""
Abstract base class and uniform protocol for all cardiovascular ML models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseCardioModel(ABC):
    """
    Standard model contract for cardiovascular disease risk prediction.
    All models must implement fit, predict, predict_proba, and evaluate.
    """

    def __init__(self, name: str, config: Any = None):
        self.name = name
        self.config = config
        self.model: Optional[BaseEstimator] = None
        self.feature_names_in_: List[str] = []
        self.is_fitted_: bool = False

    def _prepare_input(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Standardize input into numpy ndarray and capture feature names."""
        if isinstance(X, pd.DataFrame):
            if not self.is_fitted_ or not self.feature_names_in_:
                self.feature_names_in_ = list(X.columns)
                return X.to_numpy(dtype=np.float64)
            missing = [c for c in self.feature_names_in_ if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Input DataFrame is missing required features expected by '{self.name}': {missing}"
                )
            return X[self.feature_names_in_].to_numpy(dtype=np.float64)

        X_arr = np.asarray(X, dtype=np.float64)
        if not self.is_fitted_ and not self.feature_names_in_:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X_arr.shape[1])]
        elif self.is_fitted_ and self.feature_names_in_:
            if X_arr.shape[1] != len(self.feature_names_in_):
                raise ValueError(
                    f"Input array has {X_arr.shape[1]} features, but '{self.name}' "
                    f"was fitted on {len(self.feature_names_in_)} features."
                )
        return X_arr

    def _prepare_labels(
        self, y: Union[pd.Series, np.ndarray, list]
    ) -> np.ndarray:
        """Standardize label array."""
        if isinstance(y, pd.Series):
            return y.to_numpy(dtype=np.int64)
        return np.asarray(y, dtype=np.int64)

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> BaseCardioModel:
        """Fit model strictly on training data."""
        pass

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return binary class predictions (0 or 1)."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted.")
        X_arr = self._prepare_input(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return predicted probabilities for both classes: shape (N, 2).
        Column 1 represents P(cardio = 1).
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted.")
        X_arr = self._prepare_input(X)
        return self.model.predict_proba(X_arr)

    def predict_risk(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return 1D array of estimated CVD risk probabilities P(cardio = 1)."""
        proba = self.predict_proba(X)
        return proba[:, 1]

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """Compute core clinical performance metrics on provided validation/test set."""
        y_true = self._prepare_labels(y)
        y_pred = self.predict(X)
        y_prob = self.predict_risk(X)

        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_prob))

        # Calculate specificity = TN / (TN + FP)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        return {
            "roc_auc": roc_auc,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "sensitivity": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "brier_score": brier,
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return parameters for this estimator following scikit-learn convention."""
        if self.model is not None and hasattr(self.model, "get_params"):
            params = self.model.get_params(deep=deep)
            params["name"] = self.name
            params["config"] = self.config
            return params
        return {"name": self.name, "config": self.config}

    def set_params(self, **params) -> BaseCardioModel:
        """Set parameters for this estimator following scikit-learn convention."""
        valid_model_params = {}
        for key, value in params.items():
            if key == "name":
                self.name = value
            elif key == "config":
                self.config = value
            else:
                valid_model_params[key] = value

        if self.model is not None and hasattr(self.model, "set_params") and valid_model_params:
            self.model.set_params(**valid_model_params)
        return self

