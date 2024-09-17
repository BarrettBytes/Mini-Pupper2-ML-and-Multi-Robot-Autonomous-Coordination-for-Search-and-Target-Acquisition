"""
To run this script:
1. Ensure you have all required dependencies installed (torch, matplotlib, numpy)
2. Place this script in the same directory as your Demo3.py file
3. Run the script using one of the following commands:
   - For a normal run (clears previous data but not expert data db): 
        python testing_script_gen_plots.py --keep-expert-data
   - To resume from last checkpoint: 
        python testing_script_gen_plots.py --resume
   - For first run ever (clears previous data): 
        python testing_script_gen_plots.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Demo3 import ExpertPolicy, NegativeExpertPolicy, RandomPolicy, RealisticExpertPolicy, SQLiteDatabaseHandler, SimulationGridEnvironment, TrainingModel, curriculum_training, DualNetworkPolicy, ACTIONS, SEQUENCE_LENGTH, train_short_context_network
import argparse
import sqlite3
import traceback
import random
import time
import os
import time
import subprocess
import signal

# Define the directory for storing files
OUTPUT_DIR = 'testing_eval_files'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'training_progress.txt')
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'training_results.txt')
EXPERT_DATA_PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'expert_data_progress.txt')
TRAINING_DB_PATH = os.path.join(OUTPUT_DIR, 'training_data.db')

def generate_stage_id(curriculum_iter, regular_iter, second_phase_iter):
    return f"C{curriculum_iter}_R{regular_iter}_S{second_phase_iter}"

def save_progress(stage_id):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(stage_id)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return f.read().strip()
    return None

def clear_checkpoints_and_results():
    for file in os.listdir():
        if file.startswith("checkpoint_") and file.endswith(".pth"):
            os.remove(file)
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    if os.path.exists(EXPERT_DATA_PROGRESS_FILE):
        os.remove(EXPERT_DATA_PROGRESS_FILE)
    print("Cleared all checkpoints, progress file, and results file.")

def plot_results():
    if not os.path.exists(RESULTS_FILE):
        print("No results file found. Cannot plot results.")
        return

    results = {}
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            if len(line.strip().split(',')) == 3:
                stage_id, performance, dog_prob = line.strip().split(',')
            else:
                stage_id, performance = line.strip().split(',')
                dog_prob = None
            results[stage_id] = (float(performance.strip('()[]{}<>')), float(dog_prob.strip('()[]{}<>')) if dog_prob else None)

    iterations = [0, 10, 1000, 10000]
    x, y, z = np.meshgrid(iterations, iterations, iterations)
    performance = np.zeros_like(x, dtype=float)

    for i, curriculum_iter in enumerate(iterations):
        for j, regular_iter in enumerate(iterations):
            for k, second_phase_iter in enumerate(iterations):
                stage_id = generate_stage_id(curriculum_iter, regular_iter, second_phase_iter)
                performance[j, i, k] = results.get(stage_id, (0,))[0]  # Default to 0 if result not found

    # Original 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=performance, cmap='viridis')
    ax.set_xlabel('Curriculum Iterations')
    ax.set_ylabel('Regular Iterations')
    ax.set_zlabel('Second Phase Iterations')
    ax.set_title('Training Performance (Probability of Reaching Cat)')
    fig.colorbar(scatter, label='Probability')
    plt.savefig('training_results_3d_plot.png')
    plt.close()
    print("3D plot of results saved as 'training_results_3d_plot.png'")

    # New bar plots
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    labels = [item[0] for item in sorted_results]
    cat_probs = [item[1][0] for item in sorted_results]
    dog_probs = [item[1][1] for item in sorted_results if item[1][1] is not None]

    n_dogs = len(dog_probs)

    # Combined plot
    plt.figure(figsize=(15, 10))
    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x, cat_probs, width, label='Cats', color='blue')
    if n_dogs > 0:
        plt.bar(x[:n_dogs] + width, dog_probs, width, label='Dogs', color='red')

    plt.ylabel('Probability')
    plt.title('Cat and Dog Probabilities by Test')
    plt.xticks(x + width / 2, labels, rotation=90, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_probabilities.png')
    plt.close()
    print("Combined probabilities plot saved as 'combined_probabilities.png'")

    # Cats only plot
    plt.figure(figsize=(15, 10))
    plt.bar(x, cat_probs, width, color='blue')
    plt.ylabel('Probability')
    plt.title('Cat Probabilities by Test')
    plt.xticks(x, labels, rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig('cat_probabilities.png')
    plt.close()
    print("Cat probabilities plot saved as 'cat_probabilities.png'")

    # Dogs only plot (if dog probabilities exist)
    if n_dogs > 0:
        plt.figure(figsize=(15, 10))
        plt.bar(x[:n_dogs], dog_probs, width, color='red')
        plt.ylabel('Probability')
        plt.title('Dog Probabilities by Test')
        plt.xticks(x[:n_dogs], labels[:n_dogs], rotation=90, ha='right')
        plt.tight_layout()
        plt.savefig('dog_probabilities.png')
        plt.close()
        print("Dog probabilities plot saved as 'dog_probabilities.png'")


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
    
    return 1 if env.cats_reached >= cats_needed else 0, dogs_encountered


def evaluate_model(model, env):
    num_episodes = 1000
    cats_reached = 0
    total_dogs_encountered = 0
    
    for episode in range(num_episodes):
        if episode == 50 and cats_reached == 0:
            return 0, 0  # Return both probabilities as 0 if model performs poorly
        
        if episode % 50 == 0:
            print(f"evaluating episode {episode}")
        episode_result, dogs_encountered = evaluate_single_episode(env, model.policy, cats_needed=1, cats_count=1, dogs_count=1)
        cats_reached += episode_result
        total_dogs_encountered += dogs_encountered
    
    prob_cats = cats_reached / num_episodes
    prob_dogs = total_dogs_encountered / num_episodes
    return prob_cats, prob_dogs

import sys

def start_watchdog(progress_file, script_command):
    watchdog_pid = os.getpid()
    script_path = sys.argv[0]  # Get the path of the current script
    print("Starting watchdog for second phase training...")
    monitor_process = subprocess.Popen([
        "python3", "watchdog.py",
        progress_file,
        script_path,  # Pass the script path instead of the command
        *script_command[1:],  # Pass any additional arguments
        str(watchdog_pid)
    ])
    return monitor_process

def save_second_phase_progress(stage_id, episode):
                        progress_file = f"{stage_id}_second_phase_progress.txt"
                        with open(progress_file, 'w') as f:
                            f.write(str(episode))

def load_second_phase_progress(stage_id):
        progress_file = f"{stage_id}_second_phase_progress.txt"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return int(f.read().strip())
        return 0
    


def save_result(stage_id, performance, prob_dogs=None):
    with open(RESULTS_FILE, 'a') as f:
        if prob_dogs is not None:
            f.write(f"{stage_id},{performance},{prob_dogs}\n")
        else:
            f.write(f"{stage_id},{performance}\n")

def load_existing_results():
    existing_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    stage_id = parts[0]
                    # Try to clean up and convert the performance part to float
                    try:
                        performance = float(parts[1].strip('()[]{}<>'))
                    except ValueError as e:
                        print(f"Error parsing performance value '{parts[1]}' on line: {line} - {e}")
                        continue
                    
                    prob_dogs = float(parts[2].strip('()[]{}<>')) if len(parts) > 2 else None
                    existing_results[stage_id] = (performance, prob_dogs)
                else:
                    print(f"Warning: Malformed line in results file: {line}")
    return existing_results





def generate_policies(resume=False, keep_expert_data=False):
    if not resume:
        clear_checkpoints_and_results()

        if not keep_expert_data:
            if os.path.exists(TRAINING_DB_PATH):
                os.remove(TRAINING_DB_PATH)
            print("Cleared existing database.")

        if os.path.exists('base_model.pth'):
            os.remove('base_model.pth')
        print("Removed existing base model.")

    if keep_expert_data:
        print("Keeping existing expert data.")

    iterations = [0, 1000, 10000]
    env = SimulationGridEnvironment(size=6)

    base_model_path = 'base_model.pth'
    if os.path.exists(base_model_path):
        base_model = TrainingModel(env)
        base_model.set_db(TRAINING_DB_PATH)
        base_model.policy.load_state_dict(torch.load(base_model_path))
        print("Loaded existing base model.")
    else:
        print("Generating new base model with expert data...")
        base_model = altered_make_model_from_expert_data(env, num_sequences=10000, load_previous=keep_expert_data, keep_expert_data=keep_expert_data)
        torch.save(base_model.policy.state_dict(), base_model_path)
        print("Created and saved new base model.")

    # Verify that the database contains data
    db_handler = SQLiteDatabaseHandler(TRAINING_DB_PATH)
    count = db_handler.get_expert_data_count()
    if count == 0:
        raise ValueError("Database is empty after base model creation.")
    print(f"Verified: Base model created and database contains {count} records.")

    del base_model
    # Load existing results to check which configurations have already been completed
    existing_results = load_existing_results()

    for short_context in [False, True]:
        for curriculum_iter in iterations:
            for regular_iter in iterations:
                for second_phase_iter in iterations:
                    if sum(1 for i in [curriculum_iter, regular_iter, second_phase_iter] if i == 10000) > 1:
                        continue

                    second_phase_iter = second_phase_iter / 100

                    original_stage_id = generate_stage_id(curriculum_iter, regular_iter, second_phase_iter)
                    stage_id = f"SC_{original_stage_id}" if short_context else original_stage_id
                    model_name = f"model_{stage_id}.pth"
                    checkpoint_name = f"checkpoint_{stage_id}.pth"
                    temp_file = f"{stage_id}_training_complete.tmp"

                    # Check if results already exist for this configuration
                    if stage_id in existing_results:
                        print(f"Results already exist for {stage_id}. Skipping this configuration.")
                        continue

                    print(f"\nStarting stage: {stage_id}")
                    save_progress(stage_id)

                    # Load or Train the Curriculum Stage
                    curriculum_model_name = f"curriculum_{curriculum_iter}.pth"
                    if curriculum_iter > 0:
                        if os.path.exists(curriculum_model_name):
                            print(f"Loading saved curriculum model for {curriculum_iter} iterations.")
                            model = TrainingModel(env)
                            model.set_db(TRAINING_DB_PATH)
                            model.policy.load_state_dict(torch.load(curriculum_model_name))
                        else:
                            print(f"Starting curriculum training for {curriculum_iter} iterations...")
                            model = TrainingModel(env)
                            model.set_db(TRAINING_DB_PATH)
                            curriculum_training(model, num_iterations=curriculum_iter)
                            torch.save(model.policy.state_dict(), curriculum_model_name)
                            print(f"Saved curriculum model for {curriculum_iter} iterations.")
                    else:
                        model = TrainingModel(env)
                        model.set_db(TRAINING_DB_PATH)
                        model.policy.load_state_dict(torch.load(base_model_path))
                      
                    retry_stage = False

                    resume_phase = "curriculum"  # Default initialization

                    if resume:
                        if os.path.exists(temp_file):
                            print(f"Resuming from evaluation for {stage_id}")
                            model.policy.load_state_dict(torch.load(model_name))
                        elif os.path.exists(checkpoint_name):
                            print(f"Resuming training for {model_name} from checkpoint.")
                            model.policy.load_state_dict(torch.load(checkpoint_name))
                            resume_phase = determine_resume_phase(stage_id)
                            print(f"Resuming from {resume_phase} phase")


                    start_stage = None  # Reset so we continue with the next stages
                    print("training continue")

                    try:
                        # Regular training
                        model.set_db(TRAINING_DB_PATH)
                        if regular_iter > 0 and not os.path.exists(temp_file) and resume_phase == "regular":
                            print(f"Starting regular training for {regular_iter} iterations...")
                            model.train(num_iterations=regular_iter, batch_size=16)
                            torch.save(model.policy.state_dict(), checkpoint_name)
                            resume_phase = "short_context"

                        # Short context network training
                        if short_context and not(resume_phase == "second_phase"):
                            print(f"Starting short context network training for 15000 iterations...")
                            train_short_context_network(model, env, batch_size=64)
                            torch.save(model.policy.state_dict(), checkpoint_name)
                            resume_phase = "second_phase"

                        # Second phase training
                        if second_phase_iter > 0 and not os.path.exists(temp_file):
                            print(f"Starting second phase training for {second_phase_iter} iterations...")

                            progress_file = f"{stage_id}_second_phase_progress.txt"
                            script_command = ["python3", "testing_script_gen_plots.py", "--resume"]
                            watchdog_process = start_watchdog(progress_file, script_command)

                            start_episode = load_second_phase_progress(stage_id)
                            for episode in range(start_episode, int(second_phase_iter)):
                                try:
                                    model.train(num_iterations=0, batch_size=16, second_phase=True, second_phase_iterations=1, second_phase_max_attempts=10, second_phase_move_count=100, visualise_phase_two=False)
                                    torch.save(model.policy.state_dict(), checkpoint_name)
                                    save_second_phase_progress(stage_id, episode + 1)
                                    save_progress(stage_id)
                                    print(f"Completed and saved episode {episode + 1}/{int(second_phase_iter)} of second phase training.")
                                except Exception as e:
                                    print(f"Error during second phase training at episode {episode + 1}: {e}")
                                    torch.save(model.policy.state_dict(), checkpoint_name)
                                    break

                            if watchdog_process:
                                watchdog_process.terminate()

                        # Save the final model
                        torch.save(model.policy.state_dict(), model_name)
                        print(f"Completed and saved {model_name}")
                        if not os.path.exists(temp_file):
                            with open(temp_file, "w") as f:
                                f.write("Training complete")

                        # **Important**: Immediately evaluate the model and save the result
                        print(f"Evaluating model for {stage_id}...")
                        prob_cats, prob_dogs = evaluate_model(model, env)
                        print(f"Saving results for {stage_id} with performance {prob_cats} and dog probability {prob_dogs}...")
                        save_result(stage_id, prob_cats, prob_dogs)
                        print(f"Model performance for {stage_id}: Cat probability {prob_cats}, Dog probability {prob_dogs} saved successfully.")


                        # Mark results as existing for future checks
                        existing_results[stage_id] = (prob_cats, prob_dogs)

                        # Remove the checkpoint file and temporary result file
                        if os.path.exists(checkpoint_name):
                            os.remove(checkpoint_name)
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                    except sqlite3.OperationalError as e:
                        if 'no such table: expert_data' in str(e):
                            print(f"Detected missing table error for {stage_id}. Recreating the database and retrying the stage...")
                            if os.path.exists(TRAINING_DB_PATH):
                                os.remove(TRAINING_DB_PATH)  # Remove the corrupted DB
                            retry_stage = True
                        else:
                            print(f"Error occurred while processing {model_name}: {e}")
                            print("Saving checkpoint and moving to next configuration.")
                            torch.save(model.policy.state_dict(), checkpoint_name)
                    except Exception as e:
                        print(f"Error occurred while processing {model_name}: {e}")
                        print("Saving checkpoint and moving to next configuration.")
                        torch.save(model.policy.state_dict(), checkpoint_name)

                    if retry_stage:
                        # Retry the stage by resetting the model and clearing the checkpoints
                        print(f"Retrying stage {stage_id} after recreating the database...")
                        if os.path.exists(checkpoint_name):
                            os.remove(checkpoint_name)
                        retry_stage_func(stage_id, curriculum_iter, regular_iter, second_phase_iter, env, keep_expert_data)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    print("All configurations completed.")
    
    plot_results()



def retry_stage_func(stage_id, curriculum_iter, regular_iter, second_phase_iter, env, keep_expert_data):
    print(f"Retrying stage {stage_id}.")
    
    db_handler = SQLiteDatabaseHandler(TRAINING_DB_PATH)
    count = db_handler.get_expert_data_count()
    
    if count == 0:
        print("Database is empty. Regenerating expert data.")
        generate_expert_data(env, num_sequences=10000)
    else:
        print(f"Using existing expert data. Count: {count}")
    
    try:
        model = TrainingModel(env)
        model.set_db(TRAINING_DB_PATH)
        print(f"Successfully created model for {stage_id}.")
        # **Important**: Immediately evaluate the model and save the result
        print(f"Evaluating model for {stage_id}...")
        performance = evaluate_model(model, env)
        print(f"Saving results for {stage_id} with performance {performance}...")
        save_result(stage_id, performance)
        print(f"Model performance for {stage_id}: {performance} saved successfully.")
        return model
    except Exception as e:
        print(f"Error creating model for {stage_id}: {e}")
        raise


def determine_resume_phase(stage_id):
    parts = stage_id.split('_')
    
    # Handle both SC_ prefix and non-SC_ prefix cases
    if parts[0] == "SC":
        parts = parts[1:]  # Remove the SC_ prefix
    
    curriculum_iter = int(parts[0][1:])  # Remove the 'C' and convert to int
    regular_iter = int(parts[1][1:])     # Remove the 'R' and convert to int
    second_phase_iter = float(parts[2][1:])  # Remove the 'S' and convert to float
    
    second_phase_progress = load_second_phase_progress(stage_id)
    
    if second_phase_progress > 0:
        return "second_phase"
    elif regular_iter > 0:
        return "regular"
    else:
        return "curriculum"

def safe_db_execute(db_path, query, params=()):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    except sqlite3.OperationalError as e:
        if 'no such table' in str(e):
            print(f"Error: {e}. The table might be missing or the database could be corrupt.")
            raise
        else:
            print(f"Unexpected SQLite error: {e}")
            raise
    except Exception as e:
        print(f"General error during database operation: {e}")
        raise




def generate_expert_data(env, num_sequences=10000, chunk_size=1000, db_handler=None):
    print("Generating expert data...")
    if db_handler is None:
        db_handler = SQLiteDatabaseHandler(TRAINING_DB_PATH)
    db_handler.initialize_db()
    
    count = db_handler.get_expert_data_count()
    
    if count == 0:
        print("Database is empty. Generating new expert data.")
        start_from = 0
    else:
        print(f"Database already contains {count} records. Resuming from there.")
        start_from = count
    
    while start_from < num_sequences:
        try:
            end = min(start_from + chunk_size, num_sequences)
            generate_expert_data_chunk(env, end - start_from, start_from, db_handler)
            start_from = end
            print(f"Generated data up to {start_from} sequences.")
        except Exception as e:
            print(f"Error occurred while generating expert data: {e}")
            print("Saving progress and will resume from last successful point.")
            time.sleep(10)  # Wait for 10 seconds before retrying
    
    print("Expert data generation complete.")
    return db_handler

def generate_expert_data_chunk(env, num_sequences, start_from=0, db_handler=None):
    if db_handler is None:
        db_handler = SQLiteDatabaseHandler(TRAINING_DB_PATH)
    
    realistic_expert = RealisticExpertPolicy(env)
    perfect_expert = ExpertPolicy(env)
    negative_expert = NegativeExpertPolicy(env)
    random_policy = RandomPolicy(env)
    
    episode_data = []
    for episode in range(start_from, start_from + num_sequences):
        print(f"Starting episode {episode}/{start_from + num_sequences}")
        
        try:
            scenario = random.choice([
                {"cats": 1, "dogs": 1},
                {"cats": 2, "dogs": 1},
                {"cats": 1, "dogs": 2},
            ])
            
            state = env.reset(cats_count=scenario["cats"], dogs_count=scenario["dogs"])
            done = False
            move_count = 0
            episode_data = []

            while not done and move_count < 100:
                if move_count % 20 == 0:
                    print(f"Episode {episode}, Move {move_count}")

                policy_choice = random.random()
                if policy_choice < 0.7:
                    action = realistic_expert.act(state, training=True)
                elif policy_choice < 0.9:
                    action = perfect_expert.act(state, training=True)
                else:
                    action = random_policy.act(state, training=True)

                next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=scenario["cats"])
                action_index = ACTIONS.index(action)
                state_sequence = env.get_memory_state_sequence().tolist()
                next_state_sequence = next_state.tolist()
                episode_data.append((str(state_sequence), action_index, reward, str(next_state_sequence)))
                state = next_state
                move_count += 1

            print(f"Episode {episode} completed. Cats: {env.cats_reached}, Dogs: {env.dogs_reached}, Moves: {move_count}")

            # Commit data for each episode
            db_handler.insert_expert_data(episode_data)
            print(f"Committed data for episode {episode}")

        except Exception as e:
            print(f"Error in episode {episode}: {str(e)}")
            traceback.print_exc()
            continue

    print(f"Data generation chunk complete. Total episodes: {start_from + num_sequences}")




def altered_make_model_from_expert_data(env, num_sequences, load_previous=False, keep_expert_data=False):
    db_handler = SQLiteDatabaseHandler(TRAINING_DB_PATH)
    db_handler.initialize_db()

    expert_data_exists = False
    count = 0

    try:
        count = db_handler.get_expert_data_count()
        expert_data_exists = count > 0
    except Exception as e:
        print(f"Error checking expert data: {e}")
        expert_data_exists = False

    if not expert_data_exists or (not keep_expert_data and not load_previous):
        print("Generating new expert data...")
        generate_expert_data(env, num_sequences)
        count = db_handler.get_expert_data_count()
        print(f"Generated {count} expert data records.")

    if count == 0:
        raise ValueError("No expert data available after attempted generation.")

    print(f"Using {count} records for training.")

    model = TrainingModel(env)
    model.set_db(TRAINING_DB_PATH)

    # We're not training the model here, just returning it
    print("Model created with expert data.")
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training configurations")
    parser.add_argument('--resume', action='store_true', help='Resume from the last saved progress')
    parser.add_argument('--keep-expert-data', action='store_true', help='Keep existing expert data but reset everything else')
    args = parser.parse_args()

    generate_policies(resume=args.resume, keep_expert_data=args.keep_expert_data)
