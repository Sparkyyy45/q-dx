"""
Multilayer Perceptron (MLP) neural network model for cardiovascular disease risk prediction.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.config import PipelineConfig
from src.models.base import BaseCardioModel


class MLPModel(BaseCardioModel):
    """
    Multilayer Perceptron neural network with early stopping and L2 regularization.
    Outputs smooth posterior probabilities through softmax/sigmoid activation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(name="Multilayer Perceptron", config=config)
        params = config.model_params if config else None
        hidden = params.mlp_hidden_layer_sizes if params else (64, 32)
        act = params.mlp_activation if params else "relu"
        alpha = params.mlp_alpha if params else 0.001
        max_iter = params.mlp_max_iter if params else 100
        early_stop = params.mlp_early_stopping if params else True
        n_no_change = params.mlp_n_iter_no_change if params else 10
        seed = config.random_seed if config else 42

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=act,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stop,
            n_iter_no_change=n_no_change,
            random_state=seed,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> MLPModel:
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self
