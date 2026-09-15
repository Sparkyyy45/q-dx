"""
Quantum Explainability and Circuit Attribution Module.
Provides:
- Analytical parameter-shift gradient attribution across variational circuit ansatz layers
- Gate saliency analysis identifying influential quantum rotation and entangling gates
- Quantum feature attribution mapping expectation shifts back to clinical features
- Patient-level quantum state deviation analysis
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.quantum.circuit import (
    angle_encode,
    compute_parameter_shift_gradient,
    execute_variational_circuit,
)


@dataclass
class QuantumExplanation:
    """Structured attribution for quantum model predictions."""
    model_name: str
    n_qubits: int
    n_layers: int
    top_quantum_gates: List[Dict[str, Any]]
    feature_attributions: List[Dict[str, Any]]
    qubit_expectations: Dict[str, float]
    clinical_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def explain_vqc_circuit(
    model: Any,
    patient_features: Union[pd.Series, pd.DataFrame, np.ndarray],
    feature_names: List[str],
) -> QuantumExplanation:
    """
    Generate transparent circuit attribution for a fitted VQCModel on an individual patient:
    1. Computes analytical parameter-shift sensitivity for each variational rotation gate.
    2. Identifies top influential quantum gates and qubits.
    3. Evaluates Pauli-Z expectation values per qubit.
    """
    if isinstance(patient_features, pd.DataFrame):
        x_vec = patient_features.iloc[0].to_numpy(dtype=np.float64)
    elif isinstance(patient_features, pd.Series):
        x_vec = patient_features.to_numpy(dtype=np.float64)
    else:
        x_vec = np.asarray(patient_features, dtype=np.float64).flatten()

    n_qubits = getattr(model, "n_qubits", 4)
    n_layers = getattr(model, "n_layers", 2)
    parameters = getattr(model, "parameters_", None)

    if parameters is None:
        raise RuntimeError("VQC model does not contain fitted parameters.")

    x_tensor = torch.from_numpy(np.array(x_vec, copy=True)).float().unsqueeze(0)
    projection = getattr(model, "projection", None)
    if projection is not None:
        proj_features = projection(x_tensor)
        angles = angle_encode(proj_features, n_qubits=n_qubits)
    else:
        angles = angle_encode(x_tensor, n_qubits=n_qubits)

    # 1. Compute expectations
    with torch.no_grad():
        expectations, _ = execute_variational_circuit(angles, parameters, n_qubits, n_layers)
        exp_vals = expectations[0].numpy()

    qubit_expectations = {f"Qubit_{q} (Pauli-Z)": round(float(exp_vals[q]), 4) for q in range(n_qubits)}

    # 2. Compute analytical parameter-shift gradient for all variational rotation gates
    gate_attributions = []
    gate_types = ["Ry_rotation", "Rz_rotation"]

    for l in range(n_layers):
        for q in range(n_qubits):
            for p_idx in range(2):
                grad = compute_parameter_shift_gradient(
                    angles, parameters, layer=l, qubit=q, param_idx=p_idx,
                    n_qubits=n_qubits, n_layers=n_layers, observable_qubit=q,
                )
                grad_val = float(grad[0].abs().item())
                param_val = float(parameters[l, q, p_idx].item())

                gate_attributions.append({
                    "layer": l + 1,
                    "qubit": f"q_{q}",
                    "gate_type": gate_types[p_idx],
                    "parameter_angle_rad": round(param_val, 4),
                    "parameter_shift_saliency": round(grad_val, 4),
                })

    gate_attributions.sort(key=lambda x: x["parameter_shift_saliency"], reverse=True)

    # 3. Exact input-space Jacobian calculation via PyTorch autograd: d(<Z>)/d(x_j)
    x_tensor_grad = torch.from_numpy(np.array(x_vec, copy=True)).float().unsqueeze(0)
    x_tensor_grad.requires_grad_(True)
    if projection is not None:
        proj_features_grad = projection(x_tensor_grad)
        angles_grad = angle_encode(proj_features_grad, n_qubits=n_qubits)
    else:
        angles_grad = angle_encode(x_tensor_grad, n_qubits=n_qubits)

    exp_grad, _ = execute_variational_circuit(angles_grad, parameters, n_qubits, n_layers)
    loss = exp_grad.mean()
    loss.backward()
    jacobian = x_tensor_grad.grad[0].detach().numpy()

    feat_sensitivities = []
    D = len(x_vec)

    for j in range(D):
        name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
        attr_val = float(jacobian[j])
        feat_sensitivities.append({
            "feature": name,
            "patient_value": round(float(x_vec[j]), 4),
            "attribution": round(attr_val, 5),
            "direction": "increases_risk" if attr_val > 0 else "decreases_risk",
            "explanation_method": "quantum_autograd_jacobian",
        })

    feat_sensitivities.sort(key=lambda x: abs(x["attribution"]), reverse=True)

    summary = (
        f"Variational Quantum Circuit ({n_qubits} qubits, {n_layers} layers) processed patient state. "
        f"Top sensitive quantum gate: Layer {gate_attributions[0]['layer']} {gate_attributions[0]['qubit']} "
        f"({gate_attributions[0]['gate_type']}, saliency={gate_attributions[0]['parameter_shift_saliency']:.4f}). "
        f"Primary input-space clinical driver: {feat_sensitivities[0]['feature']} (attribution={feat_sensitivities[0]['attribution']:+.4f})."
    )

    return QuantumExplanation(
        model_name="Variational Quantum Classifier",
        n_qubits=n_qubits,
        n_layers=n_layers,
        top_quantum_gates=gate_attributions[:6],
        feature_attributions=feat_sensitivities,
        qubit_expectations=qubit_expectations,
        clinical_summary=summary,
    )
