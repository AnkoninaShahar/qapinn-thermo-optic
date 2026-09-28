from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

try:
    from .classical_pinn import build_mlp
except ImportError:  # pragma: no cover
    from classical_pinn import build_mlp


class QuantumFeatureLayer(nn.Module):
    """Two-qubit feature layer returning <Z0>, <Z1>.

    Q0 and Q1 use the same trainable parameter tensor. Q1 differs only by the
    CNOT after local trainable rotations.
    """

    def __init__(self, entangle: bool):
        super().__init__()
        self.entangle = entangle
        self.n_qubits = 2
        self.n_layers = 1
        self.weights = nn.Parameter(0.05 * torch.randn((self.n_layers, self.n_qubits, 3), dtype=torch.float64))
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(sample: torch.Tensor, weights: torch.Tensor):
            qml.RY(torch.pi * sample[0], wires=0)
            qml.RY(torch.pi * sample[1], wires=1)
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
                if self.entangle:
                    qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [torch.stack(self._circuit(sample, self.weights)) for sample in x]
        return torch.stack(outputs, dim=0)

    def gate_count(self) -> int:
        # RY encoding on two qubits + Rot on two qubits + optional CNOT.
        return 2 + 2 + int(self.entangle)

    def circuit_depth(self) -> int:
        # Encoding, local rotations, optional entangler.
        return 2 + int(self.entangle)


class QAPINN(nn.Module):
    """Transient QAPINN with hard BC/IC transform."""

    def __init__(self, architecture: str = "q0_separable", tail_layers: tuple[int, ...] = (16, 16)):
        super().__init__()
        if architecture in {"q0", "separable", "q0_separable"}:
            entangle = False
            self.architecture = "q0_separable"
        elif architecture in {"q1", "entangled", "q1_entangled"}:
            entangle = True
            self.architecture = "q1_entangled"
        else:
            raise ValueError("MVP supports only q0_separable and q1_entangled")
        self.quantum = QuantumFeatureLayer(entangle=entangle)
        self.classical_tail = build_mlp(2, tail_layers, 1)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum(x)

    def raw_network(self, x: torch.Tensor) -> torch.Tensor:
        return self.classical_tail(self.quantum_features(x))

    def first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x[:, 0:1]
        tau = x[:, 1:2]
        return tau * (1.0 - xi**2) * self.raw_network(x)

    def quantum_parameter_count(self) -> int:
        return sum(param.numel() for param in self.quantum.parameters())

    def gate_count(self) -> int:
        return self.quantum.gate_count()

    def circuit_depth(self) -> int:
        return self.quantum.circuit_depth()
