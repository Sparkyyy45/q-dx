"""
Quantum Support Vector Machine (QSVM) for Cardiovascular Disease Risk Prediction.
Employs quantum state fidelity kernels:
- Non-linear ZZ-entangled quantum feature mapping into 2^N dimensional Hilbert space
- Mercer-compliant Gram matrix estimation K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
- Dual SVM optimization with Platt sigmoid calibration
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PipelineConfig
from src.models.base import BaseCardioModel
from src.quantum.circuit import QuantumCircuit, compute_quantum_kernel


class QSVMModel(BaseCardioModel):
    """
    Quantum Support Vector Machine (QSVM) using quantum state fidelity kernels.
    Integrates unsupervised StandardScaler -> PCA(4) feature projection fitted strictly on
    training data, with 3-fold out-of-fold Platt scaling for calibrated probabilities.
    Conforms strictly to BaseCardioModel uniform protocol.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        n_qubits: int = 4,
        c_val: float = 1.0,
        max_train_samples: int = 1000,
        random_seed: int = 42,
    ):
        super().__init__(name="Quantum Support Vector Machine", config=config)
        self.n_qubits = n_qubits
        self.c_val = c_val
        self.max_train_samples = max_train_samples
        self.random_seed = config.random_seed if config else random_seed

        self.support_vectors_proj_: Optional[np.ndarray] = None
        self.pre_projection: Optional[Pipeline] = None
        self.model: Optional[SVC] = None
        self.calibrator_: Optional[LogisticRegression] = None
        self.circuit_ = QuantumCircuit(n_qubits=self.n_qubits, n_layers=1)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> QSVMModel:
        """
        Fit unsupervised StandardScaler -> PCA projection on X_train,
        compute quantum kernel Gram matrix, fit support vector classifier,
        and perform 3-fold out-of-fold Platt calibration.
        """
        X_arr = self._prepare_input(X)
        y_arr = self._prepare_labels(y)

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

        # 1. Fit unsupervised StandardScaler -> PCA(4) projection strictly on training split
        self.pre_projection = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=self.n_qubits, random_state=self.random_seed)),
        ])
        X_train_proj = self.pre_projection.fit_transform(X_train)
        self.support_vectors_proj_ = X_train_proj

        # 2. Compute quantum Gram matrix K_train (N_train, N_train) on projected 4D features
        K_train = compute_quantum_kernel(X_train_proj, X_train_proj, n_qubits=self.n_qubits)

        # 3. Fit production SVM classifier
        self.model = SVC(
            C=self.c_val,
            kernel="precomputed",
            random_state=self.random_seed,
        )
        self.model.fit(K_train, y_train)

        # 4. Honest Out-of-Fold (OOF) Platt Calibration via Gram submatrices
        # Reuses blocks of K_train without duplicate quantum circuit evaluations
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
        oof_decisions = np.zeros(len(X_train), dtype=np.float64)

        for tr_idx, val_idx in skf.split(X_train, y_train):
            K_tr_sub = K_train[np.ix_(tr_idx, tr_idx)]
            K_val_sub = K_train[np.ix_(val_idx, tr_idx)]
            y_tr_sub = y_train[tr_idx]

            fold_svc = SVC(C=self.c_val, kernel="precomputed", random_state=self.random_seed)
            fold_svc.fit(K_tr_sub, y_tr_sub)
            oof_decisions[val_idx] = fold_svc.decision_function(K_val_sub)

        self.calibrator_ = LogisticRegression(C=1.0, max_iter=200, random_state=self.random_seed)
        self.calibrator_.fit(oof_decisions.reshape(-1, 1), y_train)

        self.is_fitted_ = True
        return self

    def _compute_test_kernel(self, X: np.ndarray) -> np.ndarray:
        """Compute evaluation kernel matrix K_test (N_test, N_train) on projected inputs."""
        if self.support_vectors_proj_ is None or self.pre_projection is None:
            raise RuntimeError("QSVMModel is not fitted.")
        X_proj = self.pre_projection.transform(X)
        return compute_quantum_kernel(X_proj, self.support_vectors_proj_, n_qubits=self.n_qubits)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return calibrated quantum probability estimates shape (N, 2)."""
        if not self.is_fitted_ or self.model is None or self.calibrator_ is None:
            raise RuntimeError("QSVMModel is not fitted.")

        X_arr = self._prepare_input(X)
        K_test = self._compute_test_kernel(X_arr)
        decision_test = self.model.decision_function(K_test).reshape(-1, 1)
        return self.calibrator_.predict_proba(decision_test)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return binary class predictions (0 or 1)."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.50).astype(np.int64)

    def get_circuit_summary(self) -> Dict[str, Any]:
        """Return quantum architecture metadata."""
        summary = self.circuit_.get_circuit_summary()
        summary["model_name"] = self.name
        summary["is_fitted"] = self.is_fitted_
        summary["n_support_vectors"] = len(self.support_vectors_proj_) if self.support_vectors_proj_ is not None else 0
        summary["representation_type"] = "unsupervised_standard_scaler_pca_22_to_4"
        return summary

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "c_val": self.c_val,
            "max_train_samples": self.max_train_samples,
            "random_seed": self.random_seed,
            "config": self.config,
        }

    def set_params(self, **params) -> QSVMModel:
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
