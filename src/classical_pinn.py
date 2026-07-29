import torch
import torch.nn as nn

class ClassicalPINN(nn.Module):
    def __init__(self, hidden_dim=32):
        super(ClassicalPINN, self).__init__()
        # Simple Multi-Layer Perceptron (MLP)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, xi):
        """
        Hard constraint envelope: ensures theta = 0 at boundaries xi = -1 and xi = 1.
        Envelope function: (1 - xi^2)
        """
        raw_output = self.net(xi)
        envelope = (1.0 - xi ** 2)
        return envelope * raw_output

if __name__ == "__main__":
    model = ClassicalPINN()
    test_xi = torch.tensor([[0.0], [0.5], [-0.5]], requires_grad=True)
    print("Model test output:", model(test_xi))