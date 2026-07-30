import numpy as np
import torch
from physics import gaussian_heater_source, X_MIN, X_MAX

def solve_static_finite_difference(n_points=1000, A=1.0, xi_h=0.0, sigma_h=0.2):
    """
    Solves 1D steady-state Poisson equation using Finite Difference.
    - d^2 theta / d xi^2 = S(xi)
    """
    xi = np.linspace(X_MIN, X_MAX, n_points)
    dxi = xi[1] - xi[0]
    
    # Evaluate heater source S(xi)
    S = gaussian_heater_source(torch.tensor(xi, dtype=torch.float32), A, xi_h, sigma_h).numpy()
    
    # Construct second derivative operator matrix
    main_diag = -2.0 * np.ones(n_points)
    off_diag = np.ones(n_points - 1)
    D2 = (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / (dxi ** 2)
    
    # Dirichlet Boundary Conditions: theta(-1) = 0, theta(1) = 0
    D2[0, :] = 0.0; D2[0, 0] = 1.0
    D2[-1, :] = 0.0; D2[-1, -1] = 1.0
    
    rhs = -S
    rhs[0] = 0.0
    rhs[-1] = 0.0
    
    theta = np.linalg.solve(D2, rhs)
    return xi, theta

if __name__ == "__main__":
    xi, theta = solve_static_finite_difference()
    print(f"Static reference solved successfully! Peak Temperature: {np.max(theta):.4f}")