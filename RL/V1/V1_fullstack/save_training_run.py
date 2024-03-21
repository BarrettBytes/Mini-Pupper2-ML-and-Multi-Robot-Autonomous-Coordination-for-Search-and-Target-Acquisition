import torch
import Demo2  

def main():
    # Create the environment
    env = Demo2.GridEnvironment(size=6)
    
    # Generate expert data
    expert_data = Demo2.generate_expert_data(env, num_episodes=60000)
    
    # Create the IRL model
    irl_model = Demo2.IRLModel(env, expert_data)
    print("IRL model created, about to start training")
    
    # Train the IRL model
    irl_model.train(num_iterations=60000)
    
    # Get the learned policy
    learned_policy = irl_model.get_policy()
    
    # Save the trained model
    torch.save(learned_policy.state_dict(), 'learned_policy.pth')
    print("Trained model saved as 'learned_policy.pth'")

if __name__ == "__main__":
    main()
