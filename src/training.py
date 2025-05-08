import torch
import torch.optim as optim
from torch.autograd import grad
from models import PINN
from config import device, training_config, network_config
import matplotlib.pyplot as plt
import os

def compute_losses(x, model, Pe, Da, lambda_b):
    """Compute the interior and boundary losses."""
    x.requires_grad_(True)
    u = model(x)
    
    # Compute first and second derivatives
    du_dx = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    
    # Interior residual
    residual = d2u_dx2 - Pe * du_dx + Da * u * (1.0 - u)
    R_int = torch.mean(torch.square(residual))
    
    # Boundary conditions
    R_bc = torch.square(u[0] - 0.0) + torch.square(u[-1] - 1.0)
    
    # Total loss
    total_loss = R_int + lambda_b * R_bc
    
    return R_int, R_bc, total_loss

def train_model(Pe, Da, lambda_b, save_dir=None):
    """Train the PINN for given parameters."""
    # Create model and optimizer
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'], weight_decay= network_config['l2_reg'])
    
    # Generate training data
    x = torch.linspace(0, 1, training_config['n_points'], device=device).unsqueeze(1)
    
    # For storing loss history
    R_int_history = []
    R_bc_history = []
    total_loss_history = []
    
    # Training loop
    for epoch in range(training_config['epochs']):
        optimizer.zero_grad()
        
        R_int, R_bc, total_loss = compute_losses(x, model, Pe, Da, lambda_b)
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        R_int_history.append(R_int.item())
        R_bc_history.append(R_bc.item())
        total_loss_history.append(total_loss.item())
        
        # Print progress
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{training_config["epochs"]}], '
                  f'R_int: {R_int.item():.4e}, R_bc: {R_bc.item():.4e}, '
                  f'Total Loss: {total_loss.item():.4e}')
    
    # Save loss plots
    if save_dir:
        plot_losses(R_int_history, R_bc_history, total_loss_history, Pe, Da, lambda_b, save_dir)
    
    return model, R_int_history, R_bc_history

def plot_losses(R_int_history, R_bc_history, total_loss_history, Pe, Da, lambda_b, save_dir):
    """Plot and save the loss history."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(R_int_history) + 1)
    
    plt.semilogy(epochs, R_int_history, 
                 color = 'royalblue', 
                 linestyle = '--', 
                 label='Interior Loss (R_int)')
    plt.semilogy(epochs, R_bc_history, color = 'forestgreen', 
                 linestyle = '-',
                 label='Boundary Loss (R_bc)')
    plt.semilogy(epochs, total_loss_history, color = 'crimson' ,
                 linestyle = '-.', 
                 label='Total Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Loss History (Pe={Pe}, Da={Da}, λ_b={lambda_b})')
    plt.legend(fontsize=10, framealpha=1.0)  # 增加图例透明度
    plt.grid(True, linestyle=':', alpha=0.7)  # 网格线更柔和
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    filename = f"losses_Pe{Pe}_Da{Da}_lb{lambda_b}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()