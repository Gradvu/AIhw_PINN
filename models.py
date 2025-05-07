import torch
import torch.nn as nn
from config import network_config

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        input_dim = network_config['input_dim']
        output_dim = network_config['output_dim']
        width = network_config['width']
        depth = network_config['depth']
        activation = network_config['activation']
        weight_std = network_config['weight_std']
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, width))
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            
        # Output layer
        layers.append(nn.Linear(width, output_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights(weight_std)
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Activation {activation} is not supported now")
    
    def _initialize_weights(self, std):
        for layer in self.layers:
            nn.init.uniform_(layer.weight, 0.0, std)
            nn.init.uniform_(layer.bias,0.0, std)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation for output layer
                x = self.activation(x)
        return x