import numpy as np
import torch

from src.config import ProjectConfig, load_config, validate_config
from src.physics import analytical_heat_solution_torch, gaussian_heater_source_np, gaussian_heater_source_torch
from src.reference_transient import solve_transient_crank_nicolson, validate_reference_solution


def test_config_loads_and_validates():
    config = load_config("configs/physics.yaml")
    validate_config(config)


def test_gaussian_source_numpy_and_torch_match():
    xi_np = np.linspace(-1.0, 1.0, 11)
    xi_torch = torch.tensor(xi_np, dtype=torch.float64)
    np.testing.assert_allclose(
        gaussian_heater_source_np(xi_np),
        gaussian_heater_source_torch(xi_torch).detach().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_reference_solver_satisfies_ic_and_bc():
    config = ProjectConfig()
    xi, tau, theta = solve_transient_crank_nicolson(config, n_xi=31, n_tau=31)
    validation = validate_reference_solution(xi, tau, theta)
    assert validation["passes"]


def test_analytical_heat_residual_autodiff():
    xi = torch.linspace(-0.8, 0.8, 9, dtype=torch.float64).reshape(-1, 1)
    tau = torch.linspace(0.05, 0.95, 9, dtype=torch.float64).reshape(-1, 1)
    xi.requires_grad_(True)
    tau.requires_grad_(True)
    theta = analytical_heat_solution_torch(xi, tau)
    theta_tau = torch.autograd.grad(theta, tau, torch.ones_like(theta), create_graph=True)[0]
    theta_x = torch.autograd.grad(theta, xi, torch.ones_like(theta), create_graph=True)[0]
    theta_xx = torch.autograd.grad(theta_x, xi, torch.ones_like(theta_x), create_graph=True)[0]
    assert torch.max(torch.abs(theta_tau - theta_xx)).item() < 1e-10
