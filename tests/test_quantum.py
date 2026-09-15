"""
Unit tests for quantum circuit simulation, variational circuits, parameter-shift gradients,
quantum fidelity kernels, and quantum models (VQC, QSVM, Hybrid QNN).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import PipelineConfig
from src.models import create_model, get_quantum_models
from src.models.hybrid_qnn import HybridQNNModel
from src.models.qsvm_model import QSVMModel
from src.models.vqc_model import VQCModel
from src.quantum.circuit import (
    QuantumStateSimulator,
    angle_encode,
    compute_parameter_shift_gradient,
    compute_quantum_kernel,
    execute_variational_circuit,
)
from src.quantum.explain import explain_vqc_circuit


def test_quantum_state_simulator_normalization_and_superposition():
    """Verify ground state initialization and Hadamard superposition normalization."""
    sim = QuantumStateSimulator(n_qubits=3)
    state = sim.get_zero_state(batch_size=2)

    # Norm must equal 1.0
    norm = torch.sum(torch.real(state * torch.conj(state)), dim=(1, 2, 3))
    assert torch.allclose(norm, torch.ones(2)), "Initial state norm must be 1.0"

    # Apply Hadamard on qubit 0
    state_h = sim.apply_hadamard(state, qubit=0)
    norm_h = torch.sum(torch.real(state_h * torch.conj(state_h)), dim=(1, 2, 3))
    assert torch.allclose(norm_h, torch.ones(2)), "Hadamard transformed state must preserve norm 1.0"

    # Measurement on qubit 0 should be 0.0 in equal superposition
    exp = sim.measure_pauli_z_expectations(state_h)
    assert math.isclose(float(exp[0, 0].item()), 0.0, abs_tol=1e-5)


def test_quantum_gates_cnot_and_cz():
    """Verify CNOT and CZ entangling gate transformations."""
    sim = QuantumStateSimulator(n_qubits=2)
    state = sim.get_zero_state(batch_size=1)

    # Prepare |10> by applying Rx(pi) on control qubit 0
    pi_tensor = torch.tensor([math.pi])
    state_10 = sim.apply_rx(state, qubit=0, theta=pi_tensor)

    # Apply CNOT with control=0, target=1 -> Result should be |11>
    state_11 = sim.apply_cnot(state_10, control=0, target=1)
    exp = sim.measure_pauli_z_expectations(state_11)
    # Both qubits in state |1> have Pauli-Z expectation of -1.0
    assert math.isclose(float(exp[0, 0].item()), -1.0, abs_tol=1e-4)
    assert math.isclose(float(exp[0, 1].item()), -1.0, abs_tol=1e-4)

    # Apply CZ on |11> -> imparts -1 phase factor without altering probabilities
    state_cz = sim.apply_cz(state_11, control=0, target=1)
    exp_cz = sim.measure_pauli_z_expectations(state_cz)
    assert math.isclose(float(exp_cz[0, 0].item()), -1.0, abs_tol=1e-4)
    assert math.isclose(float(exp_cz[0, 1].item()), -1.0, abs_tol=1e-4)


def test_variational_circuit_expectations_bounded():
    """Verify PQC output expectations are bounded in [-1.0, 1.0]."""
    torch.manual_seed(42)
    features = torch.randn(5, 4)
    angles = angle_encode(features, n_qubits=4)
    params = torch.randn(2, 4, 2)

    exp, _ = execute_variational_circuit(angles, params, n_qubits=4, n_layers=2)
    assert exp.shape == (5, 4)
    assert (exp >= -1.0 - 1e-5).all() and (exp <= 1.0 + 1e-5).all()


def test_parameter_shift_rule_gradient():
    """Verify analytical parameter-shift gradient matches finite difference approximation."""
    torch.manual_seed(42)
    features = torch.randn(1, 4)
    angles = angle_encode(features, n_qubits=4)
    params = torch.randn(2, 4, 2)

    ps_grad = compute_parameter_shift_gradient(
        angles, params, layer=0, qubit=1, param_idx=0, n_qubits=4, n_layers=2, observable_qubit=1
    )

    # Finite difference check: (f(theta + eps) - f(theta - eps)) / (2 * eps)
    eps = 1e-4
    params_p = params.clone()
    params_p[0, 1, 0] += eps
    exp_p, _ = execute_variational_circuit(angles, params_p, n_qubits=4, n_layers=2)

    params_m = params.clone()
    params_m[0, 1, 0] -= eps
    exp_m, _ = execute_variational_circuit(angles, params_m, n_qubits=4, n_layers=2)

    fd_grad = (exp_p[0, 1] - exp_m[0, 1]) / (2.0 * eps)
    assert math.isclose(float(ps_grad[0].item()), float(fd_grad.item()), abs_tol=1e-3)


def test_quantum_kernel_properties():
    """Verify quantum fidelity kernel satisfies Mercer properties: K(x,x)=1, symmetry, in [0, 1]."""
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [-1.0, 0.5, 2.0, -0.5],
        [0.0, 1.0, 0.0, 1.0],
    ])
    K = compute_quantum_kernel(X, X, n_qubits=4)
    assert K.shape == (3, 3)

    # Diagonal must be 1.0 (state fidelity with itself)
    for i in range(3):
        assert math.isclose(float(K[i, i]), 1.0, abs_tol=1e-4)

    # Symmetry K(x_i, x_j) == K(x_j, x_i)
    assert np.allclose(K, K.T, atol=1e-5)

    # Non-negativity and bounded in [0, 1]
    assert np.all(K >= 0.0) and np.all(K <= 1.0 + 1e-5)


def test_vqc_model_interface():
    """Verify VQCModel adheres 100% to BaseCardioModel protocol."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(30, 4), columns=["a", "b", "c", "d"])
    y_train = np.array([0] * 15 + [1] * 15)
    X_test = pd.DataFrame(np.random.randn(10, 4), columns=["a", "b", "c", "d"])
    y_test = np.array([0] * 5 + [1] * 5)

    vqc = VQCModel(n_qubits=4, n_layers=1, n_epochs=3, batch_size=16)
    vqc.fit(X_train, y_train)
    assert vqc.is_fitted_

    preds = vqc.predict(X_test)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})

    probas = vqc.predict_proba(X_test)
    assert probas.shape == (10, 2)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
    assert np.allclose(probas.sum(axis=1), 1.0)

    risk = vqc.predict_risk(X_test)
    assert risk.shape == (10,)

    metrics = vqc.evaluate(X_test, y_test)
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "accuracy" in metrics

    summary = vqc.get_circuit_summary()
    assert summary["n_qubits"] == 4
    assert summary["ansatz_type"]


def test_qsvm_model_interface():
    """Verify QSVMModel adheres 100% to BaseCardioModel protocol."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(25, 4), columns=["x1", "x2", "x3", "x4"])
    y_train = np.array([0] * 13 + [1] * 12)
    X_test = pd.DataFrame(np.random.randn(8, 4), columns=["x1", "x2", "x3", "x4"])

    qsvm = QSVMModel(n_qubits=4, c_val=1.0)
    qsvm.fit(X_train, y_train)
    assert qsvm.is_fitted_

    preds = qsvm.predict(X_test)
    assert preds.shape == (8,)

    probas = qsvm.predict_proba(X_test)
    assert probas.shape == (8, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_hybrid_qnn_interface():
    """Verify HybridQNNModel adheres 100% to BaseCardioModel protocol."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(30, 6), columns=[f"f_{i}" for i in range(6)])
    y_train = np.array([0] * 15 + [1] * 15)
    X_test = pd.DataFrame(np.random.randn(10, 6), columns=[f"f_{i}" for i in range(6)])

    qnn = HybridQNNModel(n_qubits=4, n_layers=1, n_epochs=3, batch_size=16)
    qnn.fit(X_train, y_train)
    assert qnn.is_fitted_

    preds = qnn.predict(X_test)
    assert preds.shape == (10,)

    probas = qnn.predict_proba(X_test)
    assert probas.shape == (10, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_quantum_explainability_circuit_attribution():
    """Verify parameter-shift gate saliency and feature attribution."""
    X_train = pd.DataFrame(np.random.randn(20, 4), columns=["sysBP", "chol", "age", "bmi"])
    y_train = np.array([0] * 10 + [1] * 10)

    vqc = VQCModel(n_qubits=4, n_layers=2, n_epochs=2)
    vqc.fit(X_train, y_train)

    patient = X_train.iloc[0]
    explanation = explain_vqc_circuit(vqc, patient, list(X_train.columns))

    assert explanation.model_name == "Variational Quantum Classifier"
    assert len(explanation.top_quantum_gates) > 0
    assert "parameter_shift_saliency" in explanation.top_quantum_gates[0]
    assert len(explanation.feature_attributions) == 4
    assert len(explanation.qubit_expectations) == 4
    assert explanation.clinical_summary


def test_quantum_input_feature_gradient_finite_difference():
    """
    ISSUE 13 Validation:
    Verify quantum input feature gradient via autograd approximately agrees with
    finite difference [f(x + eps*e_j) - f(x - eps*e_j)] / (2*eps) across multiple features.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    n_features = 6
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_train = pd.DataFrame(np.random.randn(20, n_features), columns=feature_names)
    y_train = np.array([0] * 10 + [1] * 10)

    vqc = VQCModel(n_qubits=4, n_layers=2, n_epochs=3, random_seed=42)
    vqc.fit(X_train, y_train)

    patient = X_train.iloc[0].copy()
    exp = explain_vqc_circuit(vqc, patient, feature_names)
    attr_map = {item["feature"]: item["attribution"] for item in exp.feature_attributions}

    # Finite difference on circuit expectation across all 6 features
    eps = 2e-3
    for j, col in enumerate(feature_names):
        p_plus = patient.copy()
        p_plus[col] += eps
        p_minus = patient.copy()
        p_minus[col] -= eps

        with torch.no_grad():
            t_p = torch.tensor(p_plus.values, dtype=torch.float32).unsqueeze(0)
            t_m = torch.tensor(p_minus.values, dtype=torch.float32).unsqueeze(0)
            proj_p = vqc.projection(t_p)
            proj_m = vqc.projection(t_m)
            ang_p = angle_encode(proj_p, n_qubits=4)
            ang_m = angle_encode(proj_m, n_qubits=4)
            exp_p, _ = execute_variational_circuit(ang_p, vqc.parameters_, 4, 2)
            exp_m, _ = execute_variational_circuit(ang_m, vqc.parameters_, 4, 2)
            fd_val = float((exp_p.mean() - exp_m.mean()).item() / (2.0 * eps))

        autograd_val = attr_map[col]
        # Verify agreement within numerical tolerance
        assert math.isclose(autograd_val, fd_val, abs_tol=1e-2), (
            f"Feature {col}: autograd ({autograd_val:.5f}) does not match finite difference ({fd_val:.5f})"
        )

