import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Network architecture
network_config = {
    'input_dim': 1,
    'output_dim': 1,
    'width': 18,
    'depth': 6,
    'activation': 'tanh',
    'weight_std': 1.0,
    'l2_reg': 1e-7
}

# Training parameters
training_config = {
    'learning_rate': 1e-4,
    'epochs': 40000,
    'lambda_b_values': [10, 300, 10000],
    'n_points': 100
}

# Problem parameters
problem_params = [
    {'Pe': 0.01, 'Da': 0.01, 'name': 'diffusion_dominated'},
    {'Pe': 20.0, 'Da': 0.01, 'name': 'convection_dominated'},
    {'Pe': 0.01, 'Da': 60.0, 'name': 'reaction_dominated'}
]