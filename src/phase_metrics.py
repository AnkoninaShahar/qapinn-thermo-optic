from __future__ import annotations

import numpy as np

try:
    from .config import OpticalConfig
except ImportError:  # pragma: no cover
    from config import OpticalConfig


def mode_weight(xi: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-((xi - center) ** 2) / (2.0 * width**2))


def weighted_temperature(theta: np.ndarray, xi: np.ndarray, center: float, width: float) -> np.ndarray:
    weights = mode_weight(xi, center, width)
    denominator = np.trapz(weights, xi)
    if denominator <= 0:
        raise ValueError("optical mode normalization is non-positive")
    if theta.ndim == 1:
        return np.asarray(np.trapz(theta * weights, xi) / denominator)
    return np.trapz(theta * weights[None, :], xi, axis=1) / denominator


def phase_history(theta: np.ndarray, xi: np.ndarray, optical: OpticalConfig, which: str) -> np.ndarray:
    if which == "active":
        center = optical.active_xi
    elif which == "victim":
        center = optical.victim_xi
    else:
        raise ValueError("which must be 'active' or 'victim'")
    return optical.phase_scale * weighted_temperature(theta, xi, center, optical.mode_width)


def crosstalk_ratio(active_phase: np.ndarray, victim_phase: np.ndarray) -> float:
    active_peak = float(np.max(np.abs(active_phase)))
    if active_peak <= 1e-15:
        return float("nan")
    return float(np.max(np.abs(victim_phase)) / active_peak)


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(ref)) ** 2)))


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    denominator = np.linalg.norm(ref)
    if denominator <= 1e-15:
        return float("nan")
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(ref)) / denominator)
