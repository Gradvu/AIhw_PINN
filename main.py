import torch
from training import train_model
from graphplot import generate_test_points, plot_solution
from config import problem_params, training_config, device
import os

def main():
    # Create directories for saving results
    loss_dir = "results/losses"
    solution_dir = "results/solutions"
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(solution_dir, exist_ok=True)
    
    # Run experiments for each parameter set and lambda_b value
    for params in problem_params:
        Pe = params['Pe']
        Da = params['Da']
        name = params['name']

        print(f"\n=== Training for {name} case (Pe={Pe}, Da={Da}) ===")

        for lambda_b in training_config['lambda_b_values']:
            print(f"\nTraining with λ_b = {lambda_b}")
            
            # Train the model
            model, _, _ = train_model(Pe, Da, lambda_b, loss_dir)
            
            # Generate and plot the solution
            x_test = generate_test_points()
            with torch.no_grad():
                u_pred = model(x_test)
            
            plot_solution(x_test, u_pred, Pe, Da, lambda_b, solution_dir)
            
            # Save the model
            model_path = os.path.join("results/models", f"model_{name}_lb{lambda_b}.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()