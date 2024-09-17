import torch
import torch.nn as nn
from Demo3 import SimulationGridEnvironment, DualNetworkPolicy, evaluate_policy, ACTIONS, SEQUENCE_LENGTH
import os
import pickle
import time
import uuid

def load_model(model_path):
    try:
        input_size = 12 + len(ACTIONS)  # 4 for camera, 4 for LIDAR, 4 for direction, and 3 for actions
        hidden_size = 64
        model = DualNetworkPolicy(input_size, hidden_size, len(ACTIONS))
        
        # Load the state dict
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def safe_evaluate_policy(env, policy, num_episodes, evaluation_id, **kwargs):
    results = []
    start_episode = 0
    checkpoint_file = f'evaluation_checkpoint_{evaluation_id}.pkl'
    
    # Try to load previous results
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            results, start_episode = pickle.load(f)
        print(f"Resuming evaluation from episode {start_episode}")
    
    try:
        for episode in range(start_episode, num_episodes):
            print(f"Evaluating Episode {episode + 1}/{num_episodes}")
            episode_result = evaluate_single_episode(env, policy, **kwargs)
            results.append(episode_result)
            
            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump((results, episode + 1), f)
            
            # Add a small delay to prevent overload
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Evaluation interrupted. Progress saved.")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
    
    # Process and return results
    return process_results(results)

def evaluate_single_episode(env, policy, cats_needed=1, **kwargs):
    state = env.reset(**kwargs)
    done = False
    move_count = 0
    dogs_encountered = 0
    
    while not done and move_count < 500:
        try:
            action = policy.select_safe_action(state, env)
            state, reward, done = env.step_reward_function(action, cats_needed=cats_needed)
            move_count += 1
            
            if env.dogs_reached > dogs_encountered:
                dogs_encountered += 1
        except Exception as e:
            print(f"Error in episode step: {e}")
            break
    
    return env.cats_reached - dogs_encountered * kwargs.get('dogs_count', 2), dogs_encountered

def process_results(results):
    if not results:
        return "No results to process"
    
    performance = [r[0] for r in results]
    dogs_encountered = [r[1] for r in results]
    
    times_score_over_one = sum(1 for perf in performance if perf >= 1)
    probability_of_cat_caught = times_score_over_one / len(performance)
    probability_of_dog_encountered = sum(1 for dogs in dogs_encountered if dogs > 0) / len(dogs_encountered)
    average_performance = sum(performance) / len(performance)
    average_dogs_encountered = sum(dogs_encountered) / len(dogs_encountered)
    
    return {
        "Probability of cat Reached": probability_of_cat_caught,
        "Probability of dog Encountered": probability_of_dog_encountered,
        "Average performance": average_performance,
        "Average dogs encountered": average_dogs_encountered
    }

def clear_evaluation_results():
    for filename in os.listdir():
        if filename.startswith("evaluation_checkpoint_") and filename.endswith(".pkl"):
            os.remove(filename)
    print("Cleared previous evaluation results.")

def main():
    # Clear previous evaluation results
    clear_evaluation_results()

    model_path = 'latest_curriculum_only_policy.pth'
    learned_policy = load_model(model_path)
    
    if learned_policy is None:
        print("Failed to load the model. Exiting.")
        return

    # Standard evaluation
    print("Evaluating loaded policy on standard 6x6 grid...")
    env = SimulationGridEnvironment(size=6)
    results = safe_evaluate_policy(env, learned_policy, num_episodes=1000, 
                                   evaluation_id='standard', cats_count=2, dogs_count=2)
    print(results)

    # Large grid evaluation
    print("\nEvaluating on a larger 20x20 grid...")
    large_env = SimulationGridEnvironment(size=20)
    results = safe_evaluate_policy(large_env, learned_policy, num_episodes=20, 
                                   evaluation_id='large', cats_needed=2, cats_count=2, dogs_count=2)
    print(results)

    # Complex scenario evaluation
    print("\nEvaluating with more cats and dogs on 20x20 grid...")
    complex_env = SimulationGridEnvironment(size=20)
    results = safe_evaluate_policy(complex_env, learned_policy, num_episodes=20, 
                                   evaluation_id='complex', cats_needed=3, cats_count=5, dogs_count=5)
    print(results)

if __name__ == "__main__":
    main()