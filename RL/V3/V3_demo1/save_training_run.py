import torch
import Demo2  

def main():
    # Create the environment
    env = Demo2.GridEnvironment(size=6)
    
    # Create the IRL model
    irl_model = Demo2.make_model_from_expert_data(env, num_sequences=200000, load_previous=True)
    print("IRL model created, about to start training")
    
    # Train the IRL model
    irl_model.train(num_iterations=100000, batch_size=16, second_phase=True, second_phase_iterations=50000)
    
    # Get the learned policy
    learned_policy = irl_model.get_policy()
    
    # Save the trained model
    torch.save(learned_policy.state_dict(), 'learned_policy.pth')
    print("Trained model saved as 'learned_policy.pth'")

if __name__ == "__main__":
    main()
