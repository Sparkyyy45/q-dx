"""
CatBoost classifier for cardiovascular disease risk prediction.
Optimized gradient-boosted decision trees with native ordered target statistics
for categorical features (gender, cholesterol, gluc), bypassing continuous numerical scaling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class CatBoostModel(BaseCardioModel):
    """
    CatBoost classifier for cardiovascular risk prediction.
    Features native handling of categorical features via ordered target statistics,
    symmetric (oblivious) decision trees, and resistance to target leakage.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        cat_features: Optional[List[str]] = None,
    ):
        super().__init__(name="CatBoost", config=config)
        params = config.model_params if config else None
        iterations = getattr(params, "cat_iterations", 200) if params else 200
        depth = getattr(params, "cat_depth", 6) if params else 6
        lr = getattr(params, "cat_learning_rate", 0.05) if params else 0.05
        seed = config.random_seed if config else 42

        # Default categorical column names to track
        self.default_cat_features: List[str] = (
            cat_features
            if cat_features is not None
            else getattr(params, "cat_features", ["gender", "cholesterol", "gluc"])
            if params
            else ["gender", "cholesterol", "gluc"]
        )
        self.cat_feature_names_: List[str] = []
        self.cat_feature_indices_: List[int] = []

        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            random_seed=seed,
            verbose=0,
            thread_count=-1,
        )

    def _prepare_catboost_data(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], List[int]]:
        """
        Prepare input features specifically for CatBoost:
        1. Identifies categorical columns (gender, cholesterol, gluc).
        2. Casts categorical columns to integers to satisfy CatBoost's categorical invariant.
        3. Returns formatted DataFrame and column indices of categorical features.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if not self.is_fitted_:
                self.feature_names_in_ = list(df.columns)
                self.cat_feature_names_ = [c for c in self.default_cat_features if c in df.columns]
                self.cat_feature_indices_ = [list(df.columns).index(c) for c in self.cat_feature_names_]
            else:
                if self.feature_names_in_ and all(c in df.columns for c in self.feature_names_in_):
                    df = df[self.feature_names_in_]

            for col in self.cat_feature_names_:
                if col in df.columns:
                    df[col] = df[col].round().astype(np.int64)
            return df, self.cat_feature_indices_
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if not self.is_fitted_ and not self.feature_names_in_:
                self.feature_names_in_ = [f"feature_{i}" for i in range(X_arr.shape[1])]
            if self.cat_feature_indices_:
                df = pd.DataFrame(X_arr, columns=self.feature_names_in_)
                for idx in self.cat_feature_indices_:
                    col = self.feature_names_in_[idx]
                    df[col] = df[col].round().astype(np.int64)
                return df, self.cat_feature_indices_
            return X_arr, []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> CatBoostModel:
        X_ready, cat_indices = self._prepare_catboost_data(X)
        y_arr = self._prepare_labels(y)

        if cat_indices:
            self.model.set_params(cat_features=cat_indices)

        self.model.fit(X_ready, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted.")
        X_ready, _ = self._prepare_catboost_data(X)
        preds = self.model.predict(X_ready)
        return np.asarray(preds, dtype=np.int64).flatten()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted.")
        X_ready, _ = self._prepare_catboost_data(X)
        return self.model.predict_proba(X_ready)
