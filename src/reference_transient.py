from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

try:
    from .config import ProjectConfig, load_config
    from .physics import heater_waveform_np, source_from_config_np
except ImportError:  # pragma: no cover
    from config import ProjectConfig, load_config
    from physics import heater_waveform_np, source_from_config_np


def _laplacian_tridiagonal(n_interior: int, dxi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.ones(n_interior - 1) / dxi**2
    main = -2.0 * np.ones(n_interior) / dxi**2
    upper = np.ones(n_interior - 1) / dxi**2
    return lower, main, upper


def _to_banded(lower: np.ndarray, main: np.ndarray, upper: np.ndarray) -> np.ndarray:
    banded = np.zeros((3, main.size))
    banded[0, 1:] = upper
    banded[1, :] = main
    banded[2, :-1] = lower
    return banded


def solve_transient_crank_nicolson(
    config: ProjectConfig,
    n_xi: int | None = None,
    n_tau: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve theta_tau = theta_xx + S(xi)u(tau) with zero BC/IC.

    Returns:
        xi: shape (n_xi,)
        tau: shape (n_tau,)
        theta: shape (n_tau, n_xi)
    """
    n_xi = n_xi or config.reference.n_xi
    n_tau = n_tau or config.reference.n_tau
    xi = np.linspace(config.domain.xi_min, config.domain.xi_max, n_xi)
    tau = np.linspace(config.domain.tau_min, config.domain.tau_max, n_tau)
    dxi = xi[1] - xi[0]
    dtau = tau[1] - tau[0]

    n_interior = n_xi - 2
    lower_l, main_l, upper_l = _laplacian_tridiagonal(n_interior, dxi)
    source = source_from_config_np(xi[1:-1], config.heater)
    waveform = heater_waveform_np(tau, config.heater.waveform)

    left_lower = -0.5 * dtau * lower_l
    left_main = 1.0 - 0.5 * dtau * main_l
    left_upper = -0.5 * dtau * upper_l
    left_banded = _to_banded(left_lower, left_main, left_upper)

    right_lower = 0.5 * dtau * lower_l
    right_main = 1.0 + 0.5 * dtau * main_l
    right_upper = 0.5 * dtau * upper_l

    theta = np.zeros((n_tau, n_xi), dtype=float)
    current = np.zeros(n_interior, dtype=float)

    for n in range(n_tau - 1):
        rhs = right_main * current
        rhs[1:] += right_lower * current[:-1]
        rhs[:-1] += right_upper * current[1:]
        rhs += 0.5 * dtau * source * (waveform[n] + waveform[n + 1])
        current = solve_banded((1, 1), left_banded, rhs)
        theta[n + 1, 1:-1] = current

    return xi, tau, theta


def solve_transient_finite_difference(
    n_xi: int = 81,
    n_tau: int = 81,
    A: float = 1.0,
    xi_h: float = 0.0,
    sigma_h: float = 0.2,
    switch_off_time: float | None = None,
):
    """Backward-compatible wrapper using the MVP step-input Crank-Nicolson solver."""
    config = ProjectConfig()
    if A != config.heater.amplitude or xi_h != config.heater.center or sigma_h != config.heater.width:
        from .config import HeaterConfig

        config = ProjectConfig(heater=HeaterConfig(amplitude=A, center=xi_h, width=sigma_h))
    return solve_transient_crank_nicolson(config, n_xi=n_xi, n_tau=n_tau)


def validate_reference_solution(xi: np.ndarray, tau: np.ndarray, theta: np.ndarray) -> dict[str, float | bool]:
    initial_max = float(np.max(np.abs(theta[0, :])))
    boundary_max = float(max(np.max(np.abs(theta[:, 0])), np.max(np.abs(theta[:, -1]))))
    nonnegative_min = float(np.min(theta))
    center_trace = theta[:, len(xi) // 2]
    monotonic_violations = int(np.sum(np.diff(center_trace) < -1e-10))
    return {
        "initial_max_abs": initial_max,
        "boundary_max_abs": boundary_max,
        "min_theta": nonnegative_min,
        "center_monotonic_violations": monotonic_violations,
        "passes": initial_max < 1e-12 and boundary_max < 1e-12 and nonnegative_min > -1e-10 and monotonic_violations == 0,
    }


def reference_convergence_check(config: ProjectConfig) -> dict[str, float]:
    sizes = sorted(config.reference.convergence_grid_sizes)
    if len(sizes) < 2:
        raise ValueError("need at least two convergence grid sizes")
    xi_lo, tau_lo, theta_lo = solve_transient_crank_nicolson(config, sizes[0], sizes[0])
    xi_hi, tau_hi, theta_hi = solve_transient_crank_nicolson(config, sizes[-1], sizes[-1])
    theta_hi_on_lo = theta_hi[:: (sizes[-1] - 1) // (sizes[0] - 1), :: (sizes[-1] - 1) // (sizes[0] - 1)]
    diff = theta_lo - theta_hi_on_lo[: theta_lo.shape[0], : theta_lo.shape[1]]
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(theta_hi_on_lo), 1e-12))
    return {"coarse_n": sizes[0], "fine_n": sizes[-1], "relative_l2": rel_l2}


if __name__ == "__main__":
    cfg = load_config("configs/physics.yaml")
    xi_arr, tau_arr, theta_arr = solve_transient_crank_nicolson(cfg)
    print(validate_reference_solution(xi_arr, tau_arr, theta_arr))
