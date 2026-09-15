"""
Variational Quantum Classifier (VQC) for Cardiovascular Disease Risk Prediction.
Implements a parameterized quantum circuit (PQC) with:
- Supervised linear projection (nn.Linear(n_features, n_qubits)) to compress the full clinical
  feature representation into rotation angles without silent feature loss
- Multi-layer variational ansatz (Ry-Rz single qubit rotations + circular CNOT entanglement)
- Pauli-Z expectation measurement
- Mini-batch gradient optimization with PyTorch autograd and Adam optimizer
- Honest Out-of-Fold (OOF) 3-fold Platt sigmoid calibration (zero in-sample calibration leakage)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import PipelineConfig
from src.models.base import BaseCardioModel
from src.quantum.circuit import (
    QuantumCircuit,
    angle_encode,
    execute_variational_circuit,
)


class VQCModel(BaseCardioModel):
    """
    Variational Quantum Classifier (VQC) conforming to BaseCardioModel.
    Implements task-specific supervised projection into a 4-qubit parameterized quantum
    circuit with circular entanglement, calibrated via out-of-fold Platt scaling.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_epochs: int = 12,
        lr: float = 0.05,
        batch_size: int = 64,
        max_train_samples: int = 1000,
        random_seed: int = 42,
    ):
        super().__init__(name="Variational Quantum Classifier", config=config)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples
        self.random_seed = config.random_seed if config else random_seed

        # Variation parameters and projection layer
        torch.manual_seed(self.random_seed)
        self.parameters_ = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 2, dtype=torch.float32) * 0.1
        )
        self.projection: Optional[nn.Linear] = None
        self.calibrator_: Optional[LogisticRegression] = None
        self.circuit_ = QuantumCircuit(n_qubits=self.n_qubits, n_layers=self.n_layers)

    def _forward_quantum_expectations(
        self,
        X_tensor: torch.Tensor,
        params: torch.Tensor,
        projection: Optional[nn.Linear] = None,
    ) -> torch.Tensor:
        """Execute PQC and compute expectation values for input batch via projection."""
        proj = projection if projection is not None else self.projection
        if proj is None:
            raise RuntimeError("VQC projection layer is not initialized.")

        # Supervised linear compression of all clinical variables to n_qubits
        proj_features = proj(X_tensor)
        angles = angle_encode(proj_features, n_qubits=self.n_qubits)
        expectations, _ = execute_variational_circuit(
            features=angles,
            parameters=params,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
        )
        return expectations

    def _train_circuit_and_projection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int,
    ) -> Tuple[nn.Linear, torch.Tensor]:
        """Internal optimizer loop training both the classical projection and variational parameters."""
        D = X_train.shape[1]
        projection = nn.Linear(D, self.n_qubits)
        params = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 2, dtype=torch.float32) * 0.1
        )

        optimizer = torch.optim.Adam(
            list(projection.parameters()) + [params],
            lr=self.lr,
        )

        X_t = torch.from_numpy(np.array(X_train, copy=True)).float()
        y_t = torch.from_numpy(np.array(y_train, copy=True)).float().unsqueeze(1)
        dataset_size = len(X_train)
        n_batches = math.ceil(dataset_size / self.batch_size)

        for epoch in range(n_epochs):
            perm = torch.randperm(dataset_size)
            for b in range(n_batches):
                b_idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                x_b = X_t[b_idx]
                y_b = y_t[b_idx]

                optimizer.zero_grad()
                expectations = self._forward_quantum_expectations(x_b, params, projection=projection)
                readout = torch.mean(expectations, dim=1, keepdim=True)
                pred_prob = torch.sigmoid(readout * 2.5)

                loss = nn.functional.binary_cross_entropy(pred_prob, y_b)
                loss.backward()
                optimizer.step()

        return projection, params.detach()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> VQCModel:
        """
        Train the variational quantum parameters and projection layer,
        then calibrate probability output via 3-fold Out-of-Fold (OOF) Platt scaling.
        """
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Enforce budget sample size for quantum simulation speed
        n_samples = len(X_arr)
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

        # 1. Honest Out-of-Fold (OOF) Platt Calibration via internal 3-fold cross-validation
        # Prevents in-sample calibration leakage by fitting calibrator strictly on unseen validation readouts
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
        oof_readouts = np.zeros(len(X_train), dtype=np.float32)

        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
            X_va_f = X_train[val_idx]

            fold_proj, fold_params = self._train_circuit_and_projection(
                X_tr_f, y_tr_f, n_epochs=max(4, self.n_epochs // 2)
            )

            with torch.no_grad():
                X_va_t = torch.from_numpy(np.array(X_va_f, copy=True)).float()
                exp_va = self._forward_quantum_expectations(X_va_t, fold_params, projection=fold_proj)
                oof_readouts[val_idx] = torch.mean(exp_va, dim=1).numpy()

        self.calibrator_ = LogisticRegression(C=1.0, max_iter=200, random_state=self.random_seed)
        self.calibrator_.fit(oof_readouts.reshape(-1, 1), y_train)

        # 2. Fit final production projection layer and variational angles on full X_train
        self.projection, self.parameters_ = self._train_circuit_and_projection(
            X_train, y_train, n_epochs=self.n_epochs
        )

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return calibrated quantum probability estimates shape (N, 2)."""
        if not self.is_fitted_ or self.calibrator_ is None or self.projection is None:
            raise RuntimeError("VQCModel is not fitted.")

        X_arr = self._prepare_input(X)
        X_t = torch.from_numpy(np.array(X_arr, copy=True)).float()

        batch_size = 256
        n_samples = len(X_arr)
        readouts = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_x = X_t[i : i + batch_size]
                exp = self._forward_quantum_expectations(batch_x, self.parameters_)
                mean_exp = torch.mean(exp, dim=1).numpy()
                readouts.append(mean_exp)

        readouts_arr = np.concatenate(readouts, axis=0).reshape(-1, 1)
        probas = self.calibrator_.predict_proba(readouts_arr)
        return probas

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

    def set_params(self, **params) -> VQCModel:
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
