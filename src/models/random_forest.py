"""
Random Forest classifier for cardiovascular disease risk prediction.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class RandomForestModel(BaseCardioModel):
    """
    Random Forest ensemble model.
    Exposes impurity-based feature importances and calibrated ensemble probability estimates.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(name="Random Forest", config=config)
        params = config.model_params if config else None
        n_estimators = params.rf_n_estimators if params else 100
        max_depth = params.rf_max_depth if params else 12
        min_split = params.rf_min_samples_split if params else 10
        min_leaf = params.rf_min_samples_leaf if params else 4
        n_jobs = params.rf_n_jobs if params else -1
        seed = config.random_seed if config else 42

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            n_jobs=n_jobs,
            random_state=seed,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> RandomForestModel:
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def get_feature_importances(self) -> Dict[str, float]:
        """Return Gini impurity-based feature importance dictionary."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Model is not fitted.")
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names_in_, importances)}
