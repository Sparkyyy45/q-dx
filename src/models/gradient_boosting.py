"""
Gradient Boosting classifier for cardiovascular disease risk prediction.
Uses scikit-learn's optimized histogram-based gradient boosting algorithm.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class GradientBoostingModel(BaseCardioModel):
    """
    Histogram-based Gradient Boosting classifier.
    Optimized for numerical stability, high non-linear expressiveness, and rapid inference.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(name="Gradient Boosting", config=config)
        params = config.model_params if config else None
        max_iter = params.gb_max_iter if params else 100
        lr = params.gb_learning_rate if params else 0.08
        max_depth = params.gb_max_depth if params else 6
        min_leaf = params.gb_min_samples_leaf if params else 20
        seed = config.random_seed if config else 42

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=lr,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=seed,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> GradientBoostingModel:
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self
