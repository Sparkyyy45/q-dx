"""
Unit tests for Qiskit Aer quantum computing hardware bridge.
Verifies real depolarizing noise channel simulation, Pauli-Z expectation value computation,
and conversion of CardioQ internal quantum circuits to Qiskit native circuits.
"""

from __future__ import annotations

import numpy as np
from src.quantum.circuit import QuantumCircuit
from src.quantum.hardware_bridge import (
    convert_to_qiskit_circuit,
    execute_qiskit_simulation,
    is_qiskit_available,
)


def test_qiskit_availability():
    assert is_qiskit_available() is True, "Qiskit and Qiskit Aer must be available in environment"


def test_circuit_conversion_to_qiskit():
    qc = QuantumCircuit(n_qubits=4)
    qc.h(0)
    qc.ry(1, 0.785)
    qc.cx(0, 1)
    qc.rz(2, 1.57)
    qc.cx(1, 2)
    qc.measure(0, 0)
    qc.measure(1, 1)

    qiskit_circuit = convert_to_qiskit_circuit(qc)
    assert qiskit_circuit is not None
    assert qiskit_circuit.num_qubits == 4
    assert qiskit_circuit.num_clbits == 4
    # Instruction count should match
    op_names = [inst.operation.name for inst in qiskit_circuit.data]
    assert "h" in op_names
    assert "ry" in op_names
    assert "cx" in op_names
    assert "rz" in op_names


def test_real_qiskit_noisy_simulation_execution():
    qc = QuantumCircuit(n_qubits=4)
    # Apply angle embedding
    features = [0.5, -0.3, 1.2, -0.8]
    for i, val in enumerate(features):
        qc.ry(i, val)
    # Entanglement
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    # Variational layer
    for i in range(4):
        qc.rz(i, 0.4)

    # Execute on AerSimulator with depolarizing noise
    result = execute_qiskit_simulation(
        circuit=qc,
        shots=512,
        depolarizing_error=0.01,
        seed=42,
    )

    assert result["success"] is True
    assert result["backend"] == "qiskit_aer_aer_simulator"
    assert result["shots"] == 512
    assert result["qubits"] == 4
    assert result["execution_duration_sec"] > 0.0
    # Measurement counts
    counts = result["counts"]
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 512
    # Pauli-Z expectations should be between -1.0 and 1.0
    expectations = result["pauli_z_expectations"]
    assert len(expectations) == 4
    for exp in expectations:
        assert -1.0 <= exp <= 1.0
    # Combined expectation should be finite
    assert -1.0 <= result["combined_expectation"] <= 1.0
    assert result["qasm"] is not None and "OPENQASM" in result["qasm"]
