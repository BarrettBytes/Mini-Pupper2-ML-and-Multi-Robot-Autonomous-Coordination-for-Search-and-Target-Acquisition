import torch
import torch.nn as nn
from Demo3 import SimulationGridEnvironment, DualNetworkPolicy, evaluate_policy, ACTIONS, SEQUENCE_LENGTH

def load_model(model_path):
    # Initialize the model
    input_size = 12 + len(ACTIONS)  # 4 for camera, 4 for LIDAR, 4 for direction, and 3 for actions
    hidden_size = 64
    model = DualNetworkPolicy(input_size, hidden_size, len(ACTIONS))
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def main():
    # Load the trained policy
    model_path = 'latest_curriculum_only_policy.pth'
    learned_policy = load_model(model_path)
    
    # Create the environment
    env = SimulationGridEnvironment(size=10)
    
    # Evaluate the policy
    print("Evaluating loaded policy...")
    evaluate_policy(env, learned_policy, num_episodes=50)
    
    # Evaluate on different scenarios
    print("\nEvaluating on a larger grid...")
    large_env = SimulationGridEnvironment(size=20)
    evaluate_policy(large_env, learned_policy, num_episodes=20, cats_needed=2)
    
    print("\nEvaluating with more cats and dogs...")
    complex_env = SimulationGridEnvironment(size=20)
    evaluate_policy(complex_env, learned_policy, num_episodes=20, cats_needed=3, grid_size=20, cats_count=5, dogs_count=5)

if __name__ == "__main__":
    main()