"""
Quantum circuit simulation engine and quantum machine learning components.
High-performance exact statevector simulation with angle/amplitude encoding,
parameterized variational ansatzes, Pauli expectation measurements, and
parameter-shift / PyTorch analytical gradients.
"""

from __future__ import annotations

from src.quantum.circuit import (
    QuantumCircuit,
    QuantumStateSimulator,
    angle_encode,
    compute_quantum_kernel,
    compute_parameter_shift_gradient,
)

__all__ = [
    "QuantumCircuit",
    "QuantumStateSimulator",
    "angle_encode",
    "compute_quantum_kernel",
    "compute_parameter_shift_gradient",
]
