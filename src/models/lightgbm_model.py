"""
LightGBM classifier for cardiovascular disease risk prediction.
Optimized gradient-boosted decision trees using leaf-wise tree growth for high tabular efficiency.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class LightGBMModel(BaseCardioModel):
    """
    LightGBM classifier for cardiovascular risk prediction.
    Features leaf-wise (best-first) tree expansion, histogram-based binning, and fast training.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(name="LightGBM", config=config)
        params = config.model_params if config else None
        n_estimators = getattr(params, "lgb_n_estimators", 200) if params else 200
        num_leaves = getattr(params, "lgb_num_leaves", 31) if params else 31
        lr = getattr(params, "lgb_learning_rate", 0.05) if params else 0.05
        subsample = getattr(params, "lgb_subsample", 0.8) if params else 0.8
        colsample_bytree = getattr(params, "lgb_colsample_bytree", 0.8) if params else 0.8
        seed = config.random_seed if config else 42

        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> LightGBMModel:
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self
