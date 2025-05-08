import torch
import matplotlib.pyplot as plt
import os
from config import device

def generate_test_points(n=1000):
    """Generate points for evaluating the trained model."""
    return torch.linspace(0, 1, n, device=device).unsqueeze(1)

def plot_solution(x, u, Pe, Da, lambda_b, save_dir):
    plt.figure(figsize=(8, 6))
    
    # Convert to numpy for plotting
    x_np = x.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()
    
    plt.plot(x_np, u_np, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Solution (Pe={Pe}, Da={Da}, λ_b={lambda_b})')
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    filename = f"solution_Pe{Pe}_Da{Da}_lb{lambda_b}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()