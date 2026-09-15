"""
XGBoost classifier for cardiovascular disease risk prediction.
Optimized gradient-boosted decision trees with depth-wise tree growth and regularization controls.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class XGBoostModel(BaseCardioModel):
    """
    XGBoost classifier for cardiovascular risk prediction.
    Features exact greedy split-finding, column subsampling, and shrinkage regularization.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(name="XGBoost", config=config)
        params = config.model_params if config else None
        n_estimators = getattr(params, "xgb_n_estimators", 200) if params else 200
        max_depth = getattr(params, "xgb_max_depth", 6) if params else 6
        lr = getattr(params, "xgb_learning_rate", 0.05) if params else 0.05
        subsample = getattr(params, "xgb_subsample", 0.8) if params else 0.8
        colsample_bytree = getattr(params, "xgb_colsample_bytree", 0.8) if params else 0.8
        seed = config.random_seed if config else 42

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> XGBoostModel:
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self
