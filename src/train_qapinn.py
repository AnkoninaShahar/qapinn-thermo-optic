import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from physics import X_MIN, X_MAX
from qapinn import QAPINN
from losses import compute_static_pinn_loss
from reference_static import solve_static_finite_difference

def train_static_qapinn(architecture="entangled", epochs=500, lr=0.01):
    """
    Trains the QAPINN on the 1D static Poisson heat equation.
    """
    # Initialize the quantum model (Q1: Entangled)
    model = QAPINN(architecture=architecture, n_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Dense collocation points for the PDE loss
    xi_collocation = torch.linspace(X_MIN, X_MAX, 100).unsqueeze(1)
    
    print(f"Starting QAPINN ({architecture}) training...")
    model.train()
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute PDE residual loss
        loss = compute_static_pinn_loss(model, xi_collocation)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{epochs}], PDE Loss: {loss.item():.6f}, Time: {elapsed:.1f}s")
            
    print("Training finished!")
    return model

if __name__ == "__main__":
    # Train the quantum model
    # Note: We use fewer epochs (500) because quantum layers are highly expressive 
    # but slower to simulate than classical layers.
    trained_q_model = train_static_qapinn(architecture="entangled")
    
    # Evaluate model
    trained_q_model.eval()
    xi_test = torch.linspace(X_MIN, X_MAX, 200).unsqueeze(1)
    with torch.no_grad():
        theta_pred = trained_q_model(xi_test).numpy()
        
    # Get finite difference ground truth
    xi_ref, theta_ref = solve_static_finite_difference(n_points=200)
    
    # Plot comparison
    plt.figure(figsize=(9, 4))
    plt.plot(xi_ref, theta_ref, 'k--', label="Reference Ground Truth", linewidth=2)
    plt.plot(xi_test.numpy(), theta_pred, 'b-', label="QAPINN Prediction", linewidth=2)
    plt.title("Quantum-Assisted PINN vs. Reference Static Heat Solution")
    plt.xlabel("Position $\\xi$")
    plt.ylabel("Temperature Rise $\\theta$")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.show()