import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Demo3 import ExpertPolicy, SimulationGridEnvironment, TrainingModel, SQLiteDatabaseHandler, ACTIONS
import argparse
import sqlite3

# Define directories for storing files
OUTPUT_DIR = 'extended_testing_eval_files'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define results files for two sets of evaluations
RESULTS_FILE_1_CAT = os.path.join(OUTPUT_DIR, 'results_1_cat.txt')
RESULTS_FILE_3_CATS = os.path.join(OUTPUT_DIR, 'results_3_cats.txt')

def generate_stage_id(curriculum_iter, regular_iter, second_phase_iter):
    return f"C{curriculum_iter}_R{regular_iter}_S{second_phase_iter}"

def save_result(results_file, stage_id, performance, prob_dogs, move_count):
    with open(results_file, 'a') as f:
        f.write(f"{stage_id},{performance},{prob_dogs},{move_count}\n")

def load_existing_results(results_file):
    existing_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    stage_id = parts[0]
                    performance = float(parts[1])
                    prob_dogs = float(parts[2])
                    move_count = float(parts[3])
                    existing_results[stage_id] = (performance, prob_dogs, move_count)
    return existing_results

def plot_individual_subplot(labels, values, title, ylabel, filename, color):
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(15, 6))
    plt.bar(x, values, width, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, labels, rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{title} plot saved as '{filename}'")

def plot_results(results_file, plot_name):
    if not os.path.exists(results_file):
        print(f"No results file found for {plot_name}. Cannot plot results.")
        return

    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            stage_id, performance, prob_dogs, move_count = line.strip().split(',')
            results[stage_id] = (float(performance), float(prob_dogs), float(move_count))

    labels = list(results.keys())
    performance = [results[stage_id][0] for stage_id in labels]
    prob_dogs = [results[stage_id][1] for stage_id in labels]
    move_count = [results[stage_id][2] for stage_id in labels]

    # Save individual subplots
    if "1_cat" in plot_name:
        plot_individual_subplot(labels, performance, 'Cat Probability by Stage', 'Probability', f'cat_probability_{plot_name}.png', 'blue')
    else:
        plot_individual_subplot(labels, performance, 'Average Cats Reached by Stage', 'Average Cats', f'avg_cats_{plot_name}.png', 'blue')

    plot_individual_subplot(labels, prob_dogs, 'Dog Probability by Stage', 'Probability', f'dog_probability_{plot_name}.png', 'red')
    plot_individual_subplot(labels, move_count, 'Move Count by Stage', 'Move Count', f'move_count_{plot_name}.png', 'purple')

    # Mega plot (combined plot for this set of evaluations)
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))

    # Top plot: Cat probability or average cats
    if "1_cat" in plot_name:
        axs[0].bar(np.arange(len(labels)), performance, width=0.35, color='blue', label='Cat Prob')
        axs[0].set_title('Cat Probability by Stage (1 Cat)')
    else:
        axs[0].bar(np.arange(len(labels)), performance, width=0.35, color='blue', label='Avg Cats Reached')
        axs[0].set_title('Average Cats Reached by Stage (3 Cats)')
    axs[0].set_ylabel('Probability' if "1_cat" in plot_name else 'Average Cats')
    axs[0].set_xticks(np.arange(len(labels)))
    axs[0].set_xticklabels(labels, rotation=90, ha='right')

    # Middle plot: Dog probability
    axs[1].bar(np.arange(len(labels)), prob_dogs, width=0.35, color='red', label='Dog Prob')
    axs[1].set_title('Dog Probability by Stage')
    axs[1].set_ylabel('Probability')
    axs[1].set_xticks(np.arange(len(labels)))
    axs[1].set_xticklabels(labels, rotation=90, ha='right')

    # Bottom plot: Move count
    axs[2].bar(np.arange(len(labels)), move_count, width=0.35, color='purple', label='Move Count')
    axs[2].set_title('Move Count by Stage')
    axs[2].set_ylabel('Move Count')
    axs[2].set_xticks(np.arange(len(labels)))
    axs[2].set_xticklabels(labels, rotation=90, ha='right')

    plt.tight_layout()
    plt.savefig(f'mega_plot_{plot_name}.png')
    plt.close()
    print(f"Mega plot saved as 'mega_plot_{plot_name}.png'")

def plot_combined_results():
    """ Generate a combined plot comparing 1 cat and 3 cats evaluations """
    if not (os.path.exists(RESULTS_FILE_1_CAT) and os.path.exists(RESULTS_FILE_3_CATS)):
        print("Results files for both 1 cat and 3 cats are required. Cannot generate combined plot.")
        return

    # Load results for 1 cat
    results_1_cat = {}
    with open(RESULTS_FILE_1_CAT, 'r') as f:
        for line in f:
            stage_id, performance, prob_dogs, move_count = line.strip().split(',')
            results_1_cat[stage_id] = (float(performance), float(prob_dogs), float(move_count))

    # Load results for 3 cats
    results_3_cats = {}
    with open(RESULTS_FILE_3_CATS, 'r') as f:
        for line in f:
            stage_id, performance, prob_dogs, move_count = line.strip().split(',')
            results_3_cats[stage_id] = (float(performance), float(prob_dogs), float(move_count))

    labels_1_cat = list(results_1_cat.keys())
    performance_1_cat = [results_1_cat[stage_id][0] for stage_id in labels_1_cat]
    prob_dogs_1_cat = [results_1_cat[stage_id][1] for stage_id in labels_1_cat]
    move_count_1_cat = [results_1_cat[stage_id][2] for stage_id in labels_1_cat]

    labels_3_cats = list(results_3_cats.keys())
    avg_cats_3_cats = [results_3_cats[stage_id][0] for stage_id in labels_3_cats]
    prob_dogs_3_cats = [results_3_cats[stage_id][1] for stage_id in labels_3_cats]
    move_count_3_cats = [results_3_cats[stage_id][2] for stage_id in labels_3_cats]

    # Save individual subplots for the combined plot
    plot_individual_subplot(labels_1_cat, performance_1_cat, 'Cat Probability by Stage (1 Cat)', 'Probability', 'cat_probability_combined_1_cat.png', 'blue')
    plot_individual_subplot(labels_1_cat, prob_dogs_1_cat, 'Dog Probability by Stage (1 Cat)', 'Probability', 'dog_probability_combined_1_cat.png', 'red')
    plot_individual_subplot(labels_1_cat, move_count_1_cat, 'Move Count by Stage (1 Cat)', 'Move Count', 'move_count_combined_1_cat.png', 'purple')

    plot_individual_subplot(labels_3_cats, avg_cats_3_cats, 'Average Cats Reached by Stage (3 Cats)', 'Average Cats', 'avg_cats_combined_3_cats.png', 'blue')
    plot_individual_subplot(labels_3_cats, prob_dogs_3_cats, 'Dog Probability by Stage (3 Cats)', 'Probability', 'dog_probability_combined_3_cats.png', 'red')
    plot_individual_subplot(labels_3_cats, move_count_3_cats, 'Move Count by Stage (3 Cats)', 'Move Count', 'move_count_combined_3_cats.png', 'purple')

    # Mega Mega plot (combined plot for both sets of evaluations)
    fig, axs = plt.subplots(6, 1, figsize=(15, 24))

    # First row: Cat probability (1 cat)
    axs[0].bar(np.arange(len(labels_1_cat)), performance_1_cat, width=0.35, color='blue', label='Cat Prob (1 Cat)')
    axs[0].set_title('Cat Probability by Stage (1 Cat)')
    axs[0].set_ylabel('Probability')
    axs[0].set_xticks(np.arange(len(labels_1_cat)))
    axs[0].set_xticklabels(labels_1_cat, rotation=90, ha='right')

    # Second row: Dog probability (1 cat)
    axs[1].bar(np.arange(len(labels_1_cat)), prob_dogs_1_cat, width=0.35, color='red', label='Dog Prob (1 Cat)')
    axs[1].set_title('Dog Probability by Stage (1 Cat)')
    axs[1].set_ylabel('Probability')
    axs[1].set_xticks(np.arange(len(labels_1_cat)))
    axs[1].set_xticklabels(labels_1_cat, rotation=90, ha='right')

    # Third row: Move count (1 cat)
    axs[2].bar(np.arange(len(labels_1_cat)), move_count_1_cat, width=0.35, color='purple', label='Move Count (1 Cat)')
    axs[2].set_title('Move Count by Stage (1 Cat)')
    axs[2].set_ylabel('Move Count')
    axs[2].set_xticks(np.arange(len(labels_1_cat)))
    axs[2].set_xticklabels(labels_1_cat, rotation=90, ha='right')

    # Fourth row: Average cats reached (3 cats)
    axs[3].bar(np.arange(len(labels_3_cats)), avg_cats_3_cats, width=0.35, color='blue', label='Avg Cats Reached (3 Cats)')
    axs[3].set_title('Average Cats Reached by Stage (3 Cats)')
    axs[3].set_ylabel('Average Cats')
    axs[3].set_xticks(np.arange(len(labels_3_cats)))
    axs[3].set_xticklabels(labels_3_cats, rotation=90, ha='right')

    # Fifth row: Dog probability (3 cats)
    axs[4].bar(np.arange(len(labels_3_cats)), prob_dogs_3_cats, width=0.35, color='red', label='Dog Prob (3 Cats)')
    axs[4].set_title('Dog Probability by Stage (3 Cats)')
    axs[4].set_ylabel('Probability')
    axs[4].set_xticks(np.arange(len(labels_3_cats)))
    axs[4].set_xticklabels(labels_3_cats, rotation=90, ha='right')

    # Sixth row: Move count (3 cats)
    axs[5].bar(np.arange(len(labels_3_cats)), move_count_3_cats, width=0.35, color='purple', label='Move Count (3 Cats)')
    axs[5].set_title('Move Count by Stage (3 Cats)')
    axs[5].set_ylabel('Move Count')
    axs[5].set_xticks(np.arange(len(labels_3_cats)))
    axs[5].set_xticklabels(labels_3_cats, rotation=90, ha='right')

    plt.tight_layout()
    plt.savefig('combined_mega_plot.png')
    plt.close()
    print("Combined mega plot saved as 'combined_mega_plot.png'")

def evaluate_single_episode(env, policy, cats_needed=1, dogs_count=1, episode_num=0):
    state = env.reset(cats_count=cats_needed, dogs_count=dogs_count)
    done = False
    move_count = 0
    dogs_encountered = 0
    cats_reached = 0

    while not done and move_count < 500:
        try:
            action = policy.select_safe_action(state, env)
            state, reward, done = env.step_reward_function(action, cats_needed=cats_needed)
            move_count += 1

            if env.dogs_reached > dogs_encountered:
                dogs_encountered += 1

            if env.cats_reached > cats_reached:
                cats_reached += 1

        except Exception as e:
            print(f"Error in episode {episode_num}: {e}")
            break

    return 1 if cats_reached >= cats_needed else 0, dogs_encountered, move_count, cats_reached

def evaluate_model_1_cat(model, env):
    num_episodes = 1000
    cats_reached = 0
    total_dogs_encountered = 0
    total_moves = 0

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Evaluating 1 cat - Episode {episode}/{num_episodes}")
        episode_result, dogs_encountered, move_count, _ = evaluate_single_episode(env, model.policy, cats_needed=1, dogs_count=1, episode_num=episode)
        cats_reached += episode_result
        total_dogs_encountered += dogs_encountered
        total_moves += move_count

    prob_cats = cats_reached / num_episodes
    prob_dogs = total_dogs_encountered / num_episodes
    avg_moves = total_moves / num_episodes

    return prob_cats, prob_dogs, avg_moves

def evaluate_model_3_cats(model, env):
    num_episodes = 1000
    total_cats_reached = 0
    total_dogs_encountered = 0
    total_moves = 0

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Evaluating 3 cats - Episode {episode}/{num_episodes}")
        _, dogs_encountered, move_count, cats_reached = evaluate_single_episode(env, model.policy, cats_needed=3, dogs_count=3, episode_num=episode)
        total_cats_reached += cats_reached
        total_dogs_encountered += dogs_encountered
        total_moves += move_count

    avg_cats_reached = total_cats_reached / num_episodes
    prob_dogs = total_dogs_encountered / num_episodes
    avg_moves = total_moves / num_episodes

    return avg_cats_reached, prob_dogs, avg_moves

def regenerate_policies():
    # Evaluation for 1 cat and 1 dog
    existing_results_1_cat = load_existing_results(RESULTS_FILE_1_CAT)
    env = SimulationGridEnvironment(size=10)
    iterations = [0, 1000, 10000]

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

                    if stage_id in existing_results_1_cat:
                        print(f"Results already exist for {stage_id} (1 cat). Skipping this configuration.")
                        continue

                    print(f"\nEvaluating model for stage (1 cat): {stage_id}")
                    if not os.path.exists(model_name):
                        print(f"Model {model_name} does not exist. Skipping.")
                        continue

                    model = TrainingModel(env)
                    model.policy.load_state_dict(torch.load(model_name))

                    prob_cats, prob_dogs, avg_moves = evaluate_model_1_cat(model, env)
                    save_result(RESULTS_FILE_1_CAT, stage_id, prob_cats, prob_dogs, avg_moves)
                    print(f"Model performance for {stage_id} (1 cat): Cat prob {prob_cats}, Dog prob {prob_dogs}, Avg moves {avg_moves} saved successfully.")

    # Plot results for 1 cat and 1 dog
    plot_results(RESULTS_FILE_1_CAT, '1_cat')

    # Evaluation for 3 cats and 3 dogs
    existing_results_3_cats = load_existing_results(RESULTS_FILE_3_CATS)

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

                    if stage_id in existing_results_3_cats:
                        print(f"Results already exist for {stage_id} (3 cats). Skipping this configuration.")
                        continue

                    print(f"\nEvaluating model for stage (3 cats): {stage_id}")
                    if not os.path.exists(model_name):
                        print(f"Model {model_name} does not exist. Skipping.")
                        continue

                    model = TrainingModel(env)
                    model.policy.load_state_dict(torch.load(model_name))

                    avg_cats_reached, prob_dogs, avg_moves = evaluate_model_3_cats(model, env)
                    save_result(RESULTS_FILE_3_CATS, stage_id, avg_cats_reached, prob_dogs, avg_moves)
                    print(f"Model performance for {stage_id} (3 cats): Avg cats {avg_cats_reached}, Dog prob {prob_dogs}, Avg moves {avg_moves} saved successfully.")

    # Plot results for 3 cats and 3 dogs
    plot_results(RESULTS_FILE_3_CATS, '3_cats')

    # Generate the combined plot
    plot_combined_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate and evaluate models with additional metrics")
    args = parser.parse_args()

    regenerate_policies()
