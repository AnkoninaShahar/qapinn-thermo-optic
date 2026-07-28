import numpy as np
import torch

X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

def gaussian_heater_source(xi, A=1.0, xi_h=0.0, sigma_h=0.2):
    """
    Computes the localized Gaussian microheater power distribution S(xi).
    
    Args:
        xi (torch.Tensor or np.ndarray): Spatial coordinate(s) in [-1, 1]
        A (float): Heater power intensity
        xi_h (float): Center position of the heater
        sigma_h (float): Width parameter of the heater
    """
    return A * torch.exp(-((xi - xi_h) ** 2) / (2 * (sigma_h ** 2)))

if __name__ == "__main__":
    # Quick test to verify the heater script works
    xi_test = torch.linspace(-1.0, 1.0, 5)
    source_test = gaussian_heater_source(xi_test)
    print("Test spatial coordinates (xi):", xi_test)
    print("Test heater source values S(xi):", source_test)