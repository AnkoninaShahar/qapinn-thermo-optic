from __future__ import annotations

import torch

try:
    from .config import HeaterConfig
    from .physics import heater_waveform_torch, source_from_config_torch
except ImportError:  # pragma: no cover
    from config import HeaterConfig
    from physics import heater_waveform_torch, source_from_config_torch


def transient_pde_residual(model: torch.nn.Module, points: torch.Tensor, heater: HeaterConfig) -> torch.Tensor:
    """Return residual r = theta_tau - theta_xx - S(xi)u(tau)."""
    x = points.clone().detach().requires_grad_(True)
    theta = model(x)
    grads = torch.autograd.grad(
        theta,
        x,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
        retain_graph=True,
    )[0]
    theta_tau = grads[:, 1:2]
    theta_xi = grads[:, 0:1]
    theta_xi_grad = torch.autograd.grad(
        theta_xi,
        x,
        grad_outputs=torch.ones_like(theta_xi),
        create_graph=True,
        retain_graph=True,
    )[0]
    theta_xx = theta_xi_grad[:, 0:1]
    xi = x[:, 0:1]
    tau = x[:, 1:2]
    source = source_from_config_torch(xi, heater)
    waveform = heater_waveform_torch(tau, heater.waveform)
    return theta_tau - theta_xx - source * waveform


def transient_pde_loss(model: torch.nn.Module, points: torch.Tensor, heater: HeaterConfig) -> torch.Tensor:
    residual = transient_pde_residual(model, points, heater)
    return torch.mean(residual**2)


def compute_static_pinn_loss(model, xi_collocation, A=1.0, xi_h=0.0, sigma_h=0.2):
    """Legacy static loss retained for old exploratory notebooks."""
    xi = xi_collocation.clone().detach().requires_grad_(True)
    theta_pred = model(xi)
    d_theta_d_xi = torch.autograd.grad(
        theta_pred,
        xi,
        grad_outputs=torch.ones_like(theta_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2_theta_d_xi2 = torch.autograd.grad(
        d_theta_d_xi,
        xi,
        grad_outputs=torch.ones_like(d_theta_d_xi),
        create_graph=True,
        retain_graph=True,
    )[0]
    from .physics import gaussian_heater_source_torch

    source_term = gaussian_heater_source_torch(xi, A, xi_h, sigma_h)
    return torch.mean((-d2_theta_d_xi2 - source_term) ** 2)
