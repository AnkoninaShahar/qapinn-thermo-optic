from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_layers: tuple[int, ...], output_dim: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(last_dim, width))
        layers.append(nn.Tanh())
        last_dim = width
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class TransientClassicalPINN(nn.Module):
    """Transient parameter-matched PINN with hard BC/IC transform.

    Input shape: (batch, 2), columns [xi, tau].
    Output shape: (batch, 1), normalized temperature theta_hat.
    """

    def __init__(self, hidden_layers: tuple[int, ...] = (2, 16, 16)):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.net = build_mlp(2, hidden_layers, 1)

    def raw_network(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        first_linear = self.net[0]
        return torch.tanh(first_linear(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x[:, 0:1]
        tau = x[:, 1:2]
        return tau * (1.0 - xi**2) * self.raw_network(x)


class ClassicalPINN(TransientClassicalPINN):
    """Backward-compatible alias for the MVP transient classical model."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__(hidden_layers=(hidden_dim, hidden_dim))
