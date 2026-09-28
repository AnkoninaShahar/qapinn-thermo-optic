from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

try:
    from .config import HeaterConfig, ProjectConfig
except ImportError:  # pragma: no cover - direct script fallback
    from config import HeaterConfig, ProjectConfig

X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0


def gaussian_heater_source_np(
    xi: np.ndarray,
    amplitude: float = 1.0,
    center: float = 0.0,
    width: float = 0.2,
) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    return amplitude * np.exp(-((xi - center) ** 2) / (2.0 * width**2))


def gaussian_heater_source_torch(
    xi: torch.Tensor,
    amplitude: float = 1.0,
    center: float = 0.0,
    width: float = 0.2,
) -> torch.Tensor:
    return amplitude * torch.exp(-((xi - center) ** 2) / (2.0 * width**2))


def gaussian_heater_source(xi, A: float = 1.0, xi_h: float = 0.0, sigma_h: float = 0.2):
    """Backward-compatible Gaussian heater function."""
    if isinstance(xi, torch.Tensor):
        return gaussian_heater_source_torch(xi, A, xi_h, sigma_h)
    return gaussian_heater_source_np(np.asarray(xi), A, xi_h, sigma_h)


def source_from_config_np(xi: np.ndarray, heater: HeaterConfig) -> np.ndarray:
    return gaussian_heater_source_np(xi, heater.amplitude, heater.center, heater.width)


def source_from_config_torch(xi: torch.Tensor, heater: HeaterConfig) -> torch.Tensor:
    return gaussian_heater_source_torch(xi, heater.amplitude, heater.center, heater.width)


def heater_waveform_np(tau: np.ndarray, waveform: Literal["step"] = "step") -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    if waveform != "step":
        raise ValueError(f"unsupported MVP heater waveform: {waveform}")
    return np.ones_like(tau)


def heater_waveform_torch(tau: torch.Tensor, waveform: Literal["step"] = "step") -> torch.Tensor:
    if waveform != "step":
        raise ValueError(f"unsupported MVP heater waveform: {waveform}")
    return torch.ones_like(tau)


def analytical_heat_solution_torch(xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(math.pi**2) * tau / 4.0) * torch.cos(math.pi * xi / 2.0)


def analytical_heat_solution_np(xi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.exp(-(math.pi**2) * tau / 4.0) * np.cos(math.pi * xi / 2.0)


def make_space_time_grid(config: ProjectConfig, n_xi: int, n_tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(config.domain.xi_min, config.domain.xi_max, n_xi)
    tau = np.linspace(config.domain.tau_min, config.domain.tau_max, n_tau)
    xi_grid, tau_grid = np.meshgrid(xi, tau, indexing="xy")
    return xi, tau, xi_grid, tau_grid


if __name__ == "__main__":
    xi_test = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    print("source", gaussian_heater_source_torch(xi_test))
