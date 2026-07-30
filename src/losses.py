import torch
from physics import gaussian_heater_source

def compute_static_pinn_loss(model, xi_collocation, A=1.0, xi_h=0.0, sigma_h=0.2):
    """
    Computes the PINN loss for the 1D static heat equation (Poisson equation).
    PDE: - d^2 theta / d xi^2 = S(xi)
    """
    # Ensure gradients can be tracked with respect to spatial input xi
    xi = xi_collocation.clone().detach().requires_grad_(True)
    
    # Predict temperature from the neural network
    theta_pred = model(xi)
    
    # Compute first derivative: d(theta) / d(xi)
    d_theta_d_xi = torch.autograd.grad(
        outputs=theta_pred,
        inputs=xi,
        grad_outputs=torch.ones_like(theta_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute second derivative: d^2(theta) / d(xi^2)
    d2_theta_d_xi2 = torch.autograd.grad(
        outputs=d_theta_d_xi,
        inputs=xi,
        grad_outputs=torch.ones_like(d_theta_d_xi),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Ground truth heat source term S(xi)
    source_term = gaussian_heater_source(xi, A, xi_h, sigma_h)
    
    # PDE Residual: - d^2theta/dxi^2 - S(xi) = 0
    pde_residual = -d2_theta_d_xi2 - source_term
    loss_pde = torch.mean(pde_residual ** 2)
    
    return loss_pde