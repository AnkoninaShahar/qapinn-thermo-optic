import numpy as np
import torch
from physics import gaussian_heater_source, X_MIN, X_MAX, T_MIN, T_MAX

def solve_transient_finite_difference(n_xi=200, n_tau=500, A=1.0, xi_h=0.0, sigma_h=0.2, switch_off_time=0.5):
    """
    Solves 1D transient heat equation using Implicit Backward Euler method.
    d(theta)/d(tau) = d^2(theta)/d(xi^2) + S(xi)*u(tau)
    """
    # Create spatial and temporal grids
    xi = np.linspace(X_MIN, X_MAX, n_xi)
    tau = np.linspace(T_MIN, T_MAX, n_tau)
    dxi = xi[1] - xi[0]
    dtau = tau[1] - tau[0]
    
    # Evaluate heater source S(xi)
    S = gaussian_heater_source(torch.tensor(xi, dtype=torch.float32), A, xi_h, sigma_h).numpy()
    
    # Heating profile u(tau): 1 if tau <= switch_off_time else 0
    u = np.where(tau <= switch_off_time, 1.0, 0.0)
    
    # Set up the tridiagonal matrix for the implicit solver
    # Equation: -r * theta_{i-1}^{n+1} + (1 + 2r) * theta_i^{n+1} - r * theta_{i+1}^{n+1} = theta_i^n + dtau * S_i * u^{n+1}
    r = dtau / (dxi ** 2)
    main_diag = (1.0 + 2.0 * r) * np.ones(n_xi)
    off_diag = -r * np.ones(n_xi - 1)
    
    Matrix = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    
    # Dirichlet Boundary Conditions (Edges stay at 0 rise)
    Matrix[0, :] = 0.0; Matrix[0, 0] = 1.0
    Matrix[-1, :] = 0.0; Matrix[-1, -1] = 1.0
    
    # Initialize temperature history storage
    theta_history = np.zeros((n_tau, n_xi))
    
    # Initial condition: Chip starts at 0 temperature rise
    theta_current = np.zeros(n_xi)
    
    # Time-stepping loop
    for n in range(1, n_tau):
        rhs = theta_current + dtau * S * u[n]
        
        # Apply boundary conditions to RHS
        rhs[0] = 0.0
        rhs[-1] = 0.0
        
        # Solve the linear system for the next time step
        theta_next = np.linalg.solve(Matrix, rhs)
        theta_history[n, :] = theta_next
        theta_current = theta_next
        
    return xi, tau, theta_history

if __name__ == "__main__":
    xi, tau, theta = solve_transient_finite_difference()
    print(f"Transient reference solved! Grid shape: {theta.shape}")