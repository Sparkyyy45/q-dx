"""
Logistic Regression model for cardiovascular disease risk prediction.
Exposes coefficients, intercept, and odds ratios for clinical transparency.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class LogisticRegressionModel(BaseCardioModel):
    """
    Logistic Regression classifier with L2 penalty and lbfgs solver.
    Provides odds ratio interpretations: OR = exp(beta).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(name="Logistic Regression", config=config)
        lr_params = config.model_params if config else None
        c_val = lr_params.lr_c if lr_params else 1.0
        max_iter = lr_params.lr_max_iter if lr_params else 1000
        solver = lr_params.lr_solver if lr_params else "lbfgs"
        seed = config.random_seed if config else 42

        self.model = LogisticRegression(
            C=c_val,
            max_iter=max_iter,
            solver=solver,
            random_state=seed,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> LogisticRegressionModel:
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def get_coefficients(self) -> Dict[str, float]:
        """Return standardized model coefficients per feature."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Model is not fitted.")
        coefs = self.model.coef_[0]
        return {name: float(c) for name, c in zip(self.feature_names_in_, coefs)}

    def get_odds_ratios(self) -> Dict[str, float]:
        """Return Odds Ratios: exp(beta)."""
        coefs = self.get_coefficients()
        return {name: float(np.exp(c)) for name, c in coefs.items()}

    def get_intercept(self) -> float:
        """Return model intercept."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Model is not fitted.")
        return float(self.model.intercept_[0])
