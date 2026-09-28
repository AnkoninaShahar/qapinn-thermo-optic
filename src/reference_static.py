from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

try:
    from .config import ProjectConfig
    from .physics import source_from_config_np
except ImportError:  # pragma: no cover
    from config import ProjectConfig
    from physics import source_from_config_np


def solve_static_finite_difference(
    n_points: int = 201,
    A: float = 1.0,
    xi_h: float = 0.0,
    sigma_h: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent finite-difference solve for -theta_xx = S with zero BC."""
    config = ProjectConfig()
    xi = np.linspace(config.domain.xi_min, config.domain.xi_max, n_points)
    dxi = xi[1] - xi[0]
    source = A * np.exp(-((xi[1:-1] - xi_h) ** 2) / (2.0 * sigma_h**2))
    n_interior = n_points - 2
    main = 2.0 * np.ones(n_interior) / dxi**2
    off = -1.0 * np.ones(n_interior - 1) / dxi**2
    banded = np.zeros((3, n_interior))
    banded[0, 1:] = off
    banded[1, :] = main
    banded[2, :-1] = off
    interior = solve_banded((1, 1), banded, source)
    theta = np.zeros(n_points)
    theta[1:-1] = interior
    return xi, theta


def solve_static_from_config(config: ProjectConfig, n_points: int = 201) -> tuple[np.ndarray, np.ndarray]:
    return solve_static_finite_difference(
        n_points=n_points,
        A=config.heater.amplitude,
        xi_h=config.heater.center,
        sigma_h=config.heater.width,
    )
