import numpy as np
import torch
from src.physics import gaussian_heater_source, X_MIN, X_MAX

def solve_static_finite_difference(n_points=1000, A=1.0, xi_h=0.0, sigma_h=0.2):
    """
    Solves the 1D steady-state Poisson equation using a simple finite difference method.
    d^2 theta / d xi^2 = - S(xi)
    """
    xi = np.linspace(X_MIN, X_MAX, n_points)
    dxi = xi[1] - xi[0]
    
    # Evaluate the source term S(xi) on the grid
    # (convert temporarily to numpy if using torch tensors)
    S = gaussian_heater_source(torch.tensor(xi), A, xi_h, sigma_h).numpy()
    
    # Set up a tridiagonal matrix for second derivatives (Poisson equation)
    main_diag = -2.0 * np.ones(n_points)
    off_diag = np.ones(n_points - 1)
    
    # Constructing the full finite difference operator matrix D2 / dxi^2
    D2 = (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / (dxi ** 2)
    
    # Apply Dirichlet boundary conditions (theta = 0 at boundaries)
    D2[0, :] = 0; D2[0, 0] = 1.0
    D2[-1, :] = 0; D2[-1, -1] = 1.0
    
    rhs = -S
    rhs[0] = 0.0   # Boundary condition at xi = -1
    rhs[-1] = 0.0  # Boundary condition at xi = 1
    
    # Solve linear system
    theta = np.linalg.solve(D2, rhs)
    return xi, theta

if __name__ == "__main__":
    xi, theta = solve_static_finite_difference()
    print(f"Static reference solved! Max temperature rise: {np.max(theta):.4f} K")