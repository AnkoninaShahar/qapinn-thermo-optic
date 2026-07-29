import torch
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    from src.physics import X_MIN, X_MAX
    from src.classical_pinn import ClassicalPINN
    from src.losses import compute_static_pinn_loss
    from src.reference_static import solve_static_finite_difference
except ModuleNotFoundError:
    from physics import X_MIN, X_MAX
    from classical_pinn import ClassicalPINN
    from losses import compute_static_pinn_loss
    from reference_static import solve_static_finite_difference

def train_static_pinn(epochs=5000, lr=0.003, hidden_dim=32):
    """
    Trains the classical PINN on the 1D static Poisson heat equation.
    """
    model = ClassicalPINN(hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use a dense, evenly spaced grid so the model captures the narrow heater peak at xi = 0
    xi_collocation = torch.linspace(X_MIN, X_MAX, 300).unsqueeze(1)
    
    print("Starting Classical PINN training...")
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute PDE residual loss
        loss = compute_static_pinn_loss(model, xi_collocation)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], PDE Loss: {loss.item():.6f}")
            
    print("Training finished!")
    return model

if __name__ == "__main__":
    trained_model = train_static_pinn()
    
    # Evaluate model
    trained_model.eval()
    xi_test = torch.linspace(X_MIN, X_MAX, 200).unsqueeze(1)
    with torch.no_grad():
        theta_pred = trained_model(xi_test).numpy()
        
    # Get finite difference ground truth
    xi_ref, theta_ref = solve_static_finite_difference(n_points=200)
    
    # Plot comparison
    plt.figure(figsize=(9, 4))
    plt.plot(xi_ref, theta_ref, 'k--', label="Reference Ground Truth", linewidth=2)
    plt.plot(xi_test.numpy(), theta_pred, 'r-', label="Classical PINN Prediction", linewidth=2)
    plt.title("Classical PINN vs. Reference Static Heat Solution")
    plt.xlabel("Position $\\xi$")
    plt.ylabel("Temperature Rise $\\theta$")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.show()