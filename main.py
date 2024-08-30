from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from env import GridEnv
import numpy as np
import torch

def main():

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the environment
    env = GridEnv(render_mode="human", device=device)
    
    # Reset the environment to the initial state
    env.reset()
    
    # Run the ACO algorithm
    env.run_aco()
    
    # Render the final state of the grid
    env.render()

    # Visualize pheromone levels
    env.visualize_pheromones()

if __name__ == "__main__":
    main()