import torch
import matplotlib.pyplot as plt
import random
from Demo3 import SimulationGridEnvironment, TrainingModel, DualNetworkPolicy, evaluate_policy, create_fortress


def resume_training_and_evaluation():
    print("Loading curriculum-trained model...")
    env = SimulationGridEnvironment(size=10)
    model = TrainingModel(env)
    
    # Load the saved model
    policy = DualNetworkPolicy(input_size=15, hidden_size=64, num_actions=3)
    
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the state dict and move it to the appropriate device
    state_dict = torch.load('latest_curriculum_only_policy.pth', map_location=device)
    policy.load_state_dict(state_dict)
    
    # Ensure the entire model is on the correct device
    policy = policy.to(device)
    model.policy = policy
    model.device = device  # Ensure the TrainingModel knows about the device
    
    print("Curriculum-trained model loaded, starting regular training")
    
    # Debug information
    print(f"Model device: {next(model.policy.parameters()).device}")
    
    try:
        model.train(num_iterations=10, batch_size=16, redo_short_context=False, second_phase=True, second_phase_iterations=10)
    except RuntimeError as e:
        print(f"RuntimeError occurred: {e}")
        print("Debug information:")
        print(f"Model device: {next(model.policy.parameters()).device}")
        # Add any other relevant debug information here
        return

    learned_policy = model.get_policy()

    torch.save(learned_policy.state_dict(), 'latest_policy.pth')
    print("Trained model saved as 'latest_policy.pth'")

    evaluate_policy(env, learned_policy, num_episodes=10)

    # Evaluate on the fortress scenario
    print("Evaluating on fortress scenario...")
    fortress_env = SimulationGridEnvironment(size=10)
    quadrant = random.randint(0, 6)
    fortress, cat_position = create_fortress(fortress_env, quadrant)
    fortress_env.obstacle_positions = fortress
    fortress_env.cat_positions = [cat_position]
    evaluate_policy(fortress_env, learned_policy, num_episodes=20, cats_needed=1, grid_size=10, cats_count=1, dogs_count=0)

    env = SimulationGridEnvironment(size=20)
    evaluate_policy(env, learned_policy, num_episodes=20, cats_needed=2)

    env = SimulationGridEnvironment(size=20)
    evaluate_policy(env, learned_policy, num_episodes=20, cats_needed=3, grid_size=10, cats_count=5, dogs_count=5)

    # Plot second phase training losses
    plt.figure(figsize=(10, 5))
    plt.plot(model.losses_phase_two)
    plt.title('Second Phase Training Losses')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.savefig('second_phase_training_losses.png')
    plt.close()

if __name__ == "__main__":
    resume_training_and_evaluation()