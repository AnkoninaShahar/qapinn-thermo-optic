import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

class QAPINN(nn.Module):
    def __init__(self, architecture="separable", n_layers=2, hidden_dim=32):
        """
        architecture options: "separable", "entangled", "reupload"
        """
        super(QAPINN, self).__init__()
        self.architecture = architecture
        self.n_layers = n_layers
        
        # 1. Define the Quantum Circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            """
            inputs: Spatial/Temporal variables. Shape must be (batch_size, num_features)
            weights: Trainable rotation angles for the quantum gates
            """
            num_features = inputs.shape[1]
            
            # Base Data Encoding: Convert classical data into quantum phase
            for i in range(n_qubits):
                # Safely slice the i-th column for the entire batch
                feature_col = inputs[:, i % num_features] 
                qml.RY(np.pi * feature_col, wires=i)
                
            for layer in range(self.n_layers):
                # Trainable Rotations (The "weights" of the quantum layer)
                for i in range(n_qubits):
                    qml.Rot(*weights[layer, i], wires=i)
                    
                # Quantum Entanglement (Links space and time qubits together)
                if self.architecture in ["entangled", "reupload"]:
                    qml.CNOT(wires=[0, 1])
                    
                # Data Re-uploading (Creates higher-order Fourier harmonics)
                if self.architecture == "reupload" and layer < self.n_layers - 1:
                    for i in range(n_qubits):
                        feature_col = inputs[:, i % num_features]
                        qml.RY(np.pi * feature_col, wires=i)
                        
            # Return the measured expectation values of the qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # 2. Bridge PennyLane and PyTorch
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # 3. Classical Tail
        self.classical_tail = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # Guarantee input is always 2D (batch_size, features) before quantum layer
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        # Pass inputs through the Quantum Layer
        q_out = self.q_layer(x) 
        
        # Pass quantum expectation values into the Classical Tail
        raw_out = self.classical_tail(q_out)
        
        # Hard constraint envelope: force edges to zero rise
        xi = x[:, 0:1]
        envelope = (1.0 - xi ** 2)
        return envelope * raw_out

if __name__ == "__main__":
    # Test the updated PyTorch + PennyLane bridge with batched data
    test_model = QAPINN(architecture="entangled")
    
    # 2 rows in the batch, 1 feature (spatial coordinate)
    test_inputs = torch.tensor([[0.5], [-0.2]], requires_grad=True)
    
    output = test_model(test_inputs)
    print("QAPINN Output Shape:", output.shape)
    print("QAPINN Output Values:\n", output.detach().numpy())