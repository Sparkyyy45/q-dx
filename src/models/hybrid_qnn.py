"""
Hybrid Quantum-Classical Neural Network (QNN) for Cardiovascular Risk Prediction.
Combines:
- Classical feedforward dimension compression layers
- Variational quantum circuit expectation layer (PQC)
- Classical dense readout classification head
- End-to-end backpropagation through hybrid quantum-classical boundaries
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from src.config import PipelineConfig
from src.models.base import BaseCardioModel
from src.quantum.circuit import (
    QuantumCircuit,
    execute_variational_circuit,
)


class _HybridQuantumNet(nn.Module):
    """PyTorch hybrid quantum-classical neural network module."""

    def __init__(self, in_features: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical encoder: projects arbitrary input dimensionality into n_qubits bounded angles
        self.encoder = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.Tanh(),
        )

        # Variational quantum circuit parameters: (layers, qubits, [Ry, Rz])
        self.quantum_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, dtype=torch.float32) * 0.1
        )

        # Classical decoder: maps quantum expectation values to prediction logit
        self.decoder = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Classical feature transformation to angles in [-pi, pi]
        angles = self.encoder(x) * math.pi

        # 2. Parameterized Quantum Circuit execution
        expectations, _ = execute_variational_circuit(
            features=angles,
            parameters=self.quantum_params,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
        )

        # 3. Classical classification head
        logits = self.decoder(expectations)
        return torch.sigmoid(logits)


class HybridQNNModel(BaseCardioModel):
    """
    Hybrid Quantum-Classical Neural Network (QNN) conforming to BaseCardioModel protocol.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_epochs: int = 15,
        lr: float = 0.02,
        batch_size: int = 64,
        max_train_samples: int = 4000,
        random_seed: int = 42,
    ):
        super().__init__(name="Hybrid Quantum Neural Network", config=config)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples
        self.random_seed = config.random_seed if config else random_seed

        self.net: Optional[_HybridQuantumNet] = None
        self.circuit_ = QuantumCircuit(n_qubits=self.n_qubits, n_layers=self.n_layers)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> HybridQNNModel:
        """Fit hybrid quantum-classical parameters using Adam optimizer."""
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        n_samples, in_features = X_arr.shape

        # Stratified sampling if cohort exceeds simulation budget
        if n_samples > self.max_train_samples:
            idx_0 = np.where(y_arr == 0)[0]
            idx_1 = np.where(y_arr == 1)[0]
            n_half = self.max_train_samples // 2
            sub_idx_0 = np.random.choice(idx_0, size=min(n_half, len(idx_0)), replace=False)
            sub_idx_1 = np.random.choice(idx_1, size=min(n_half, len(idx_1)), replace=False)
            train_idx = np.concatenate([sub_idx_0, sub_idx_1])
            np.random.shuffle(train_idx)
            X_train = X_arr[train_idx]
            y_train = y_arr[train_idx]
        else:
            X_train = X_arr
            y_train = y_arr

        X_t = torch.from_numpy(np.array(X_train, copy=True)).float()
        y_t = torch.from_numpy(np.array(y_train, copy=True)).float().unsqueeze(1)

        self.net = _HybridQuantumNet(
            in_features=in_features, n_qubits=self.n_qubits, n_layers=self.n_layers
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset_size = len(X_train)
        n_batches = math.ceil(dataset_size / self.batch_size)

        self.net.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(dataset_size)
            for b in range(n_batches):
                b_idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                x_b = X_t[b_idx]
                y_b = y_t[b_idx]

                optimizer.zero_grad()
                preds = self.net(x_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()

        self.net.eval()
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return predicted probability matrix shape (N, 2)."""
        if not self.is_fitted_ or self.net is None:
            raise RuntimeError("HybridQNNModel is not fitted.")

        X_arr = self._prepare_input(X)
        X_t = torch.from_numpy(np.array(X_arr, copy=True)).float()

        batch_size = 256
        n_samples = len(X_arr)
        prob_list = []

        self.net.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_x = X_t[i : i + batch_size]
                probs = self.net(batch_x).squeeze(1).numpy()
                prob_list.append(probs)

        pos_probs = np.concatenate(prob_list, axis=0)
        pos_probs = np.clip(pos_probs, 1e-6, 1.0 - 1e-6)
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return binary class predictions (0 or 1)."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.50).astype(np.int64)

    def get_circuit_summary(self) -> Dict[str, Any]:
        """Return quantum architecture metadata."""
        summary = self.circuit_.get_circuit_summary()
        summary["model_name"] = self.name
        summary["is_fitted"] = self.is_fitted_
        return summary

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_train_samples": self.max_train_samples,
            "random_seed": self.random_seed,
            "config": self.config,
        }

    def set_params(self, **params) -> HybridQNNModel:
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
