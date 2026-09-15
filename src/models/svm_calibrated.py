"""
Support Vector Machine with train-only sigmoid (Platt) calibration.
Ensures probability estimates are well-calibrated and bounded strictly in [0, 1].
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class CalibratedSVMModel(BaseCardioModel):
    """
    Support Vector Classifier with train-only Sigmoid (Platt) calibration.
    Uses LinearSVC as the underlying max-margin classifier for computational efficiency
    on large tabular cohorts, wrapped with CalibratedClassifierCV(method='sigmoid').
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(name="Calibrated SVM", config=config)
        params = config.model_params if config else None
        c_val = params.svm_c if params else 1.0
        max_iter = params.svm_max_iter if params else 2000
        cal_method = params.svm_calibration_method if params else "sigmoid"
        cal_cv = params.svm_calibration_cv if params else 3
        seed = config.random_seed if config else 42

        # Base margin classifier
        self.base_estimator = LinearSVC(
            C=c_val,
            max_iter=max_iter,
            dual="auto",
            random_state=seed,
        )

        # Sigmoid Platt calibration fitted exclusively on training partitions
        self.model = CalibratedClassifierCV(
            estimator=self.base_estimator,
            method=cal_method,
            cv=cal_cv,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> CalibratedSVMModel:
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def get_base_estimator(self) -> LinearSVC:
        """Return base SVM estimator."""
        return self.base_estimator
