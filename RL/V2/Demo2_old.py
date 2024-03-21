import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from collections import deque
from utils import load_and_resize_image, create_video_writer, add_frame_to_video, add_plot_to_video, display_environment

# Define the action space
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SEQ_LENGTH = 30  # Length of the action and sensor history

class GridEnvironment:
    def __init__(self, size=6, dog_image_path='dog.png', cat_image_path='cat.png', robot_image_path='robot.png'):
        print("Initializing environment...")
        self.size = size
        self.dog_image = load_and_resize_image(dog_image_path)
        self.cat_image = load_and_resize_image(cat_image_path)
        self.robot_image = load_and_resize_image(robot_image_path)
        self.cell_height, self.cell_width, _ = self.dog_image.shape
        self.grid_height = self.cell_height * self.size
        self.grid_width = self.cell_width * self.size
        self.reset(training=False)

    def reset(self, training=False, preserve_positions=False, cats_count=2, dogs_count=2, preserve_cats_found=False):
        # if not training:
        #     print("Resetting environment...")
        if not preserve_positions:
            self.cat_positions = self.generate_positions(cats_count) 
            self.dog_positions = self.generate_positions(dogs_count, avoid=self.cat_positions)
        if not preserve_cats_found:
            self.found_cats = [False] * len(self.cat_positions)
        self.moves = 0
        if not preserve_positions:
            self.position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        self.cats_reached = sum(self.found_cats)  # Update the number of cats reached
        self.dogs_reached = 0
        self.state_history = deque(maxlen=SEQ_LENGTH)
        self.action_history = deque(maxlen=SEQ_LENGTH)
        self.update_state()
        return self.get_state_sequence()

    def generate_positions(self, num_positions, min_distance=3, avoid=[]):
        positions = []
        while len(positions) < num_positions:
            position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if all(np.linalg.norm(np.array(position) - np.array(p)) >= min_distance for p in positions + avoid):
                positions.append(tuple(position))
        return positions

    def get_camera_data(self):
        camera_view = np.zeros(5)  # Camera view for left, right, up, down
        x, y = self.position
        
        # Check left direction
        for i in range(1, x + 1):
            if (x - i, y) in self.cat_positions or (x - i, y) in self.dog_positions:
                if (x - i, y) in self.dog_positions:
                    camera_view[0] = 3
                    break
                camera_view[0] = 2
                break
            
        # Check right direction
        for i in range(1, self.size - x):
            if (x + i, y) in self.cat_positions or (x + i, y) in self.dog_positions:
                if (x + i, y) in self.dog_positions:
                    camera_view[1] = 3
                    break
                camera_view[1] = 2
                break
        
        # Check up direction
        for i in range(1, y + 1):
            if (x, y - i) in self.cat_positions or (x, y - i) in self.dog_positions:
                if (x, y - i) in self.dog_positions:
                    camera_view[2] = 3
                    break
                camera_view[2] = 2
                break
         
        # Check down direction
        for i in range(1, self.size - y):
            if (x, y + i) in self.cat_positions or (x, y + i) in self.dog_positions:
                if (x, y + i) in self.dog_positions:
                    camera_view[3] = 3
                    break
                camera_view[3] = 2
                break

        return camera_view

    def step(self, action, training=False, cats_needed=1):
        if not self.is_move_possible(action):
            return self.get_state_sequence(), -0.05, False

        # Get camera data before moving
        camera_data_before = self.get_camera_data()

        new_position = (self.position[0] + action[0], self.position[1] + action[1])
        self.position = new_position
        self.moves += 1

        reward = -0.05
        done = False

        if self.position in self.cat_positions:
            idx = self.cat_positions.index(self.position)
            if not self.found_cats[idx]:
                self.found_cats[idx] = True
                reward = 1.0
                self.cats_reached += 1

        if self.position in self.dog_positions:
            reward = -5.0
            self.dogs_reached += 1
            done = True

        # Get camera data after moving
        camera_data_after = self.get_camera_data()
        if 3 in camera_data_before:
            if self.is_moving_towards_dog(camera_data_before, camera_data_after, action):
                reward -= 2.0

        if self.cats_reached >= cats_needed:
            done = True
            reward = 10.0

        if self.moves >= 200:
            done = True

        self.record_state(action)
        return self.get_state_sequence(), reward, done

    def is_moving_towards_dog(self, camera_data_before, camera_data_after, action):
        dog_detected_index = np.where(camera_data_before == 3)[0][0]
        x_movement, y_movement = action
        if x_movement > 0:
            return dog_detected_index > self.position[0]
        elif x_movement < 0:
            return dog_detected_index < self.position[0]
        elif y_movement > 0:
            return dog_detected_index > self.position[1]
        elif y_movement < 0:
            return dog_detected_index < self.position[1]
        return False

    def update_state(self):
        # Update the current state representation
        camera_data = self.get_camera_data()
        self.state_history.append(camera_data)
        self.action_history.append(np.zeros(len(ACTIONS)))

    def record_state(self, action):
        camera_data = self.get_camera_data()
        action_one_hot = np.zeros(len(ACTIONS))
        if action in ACTIONS:
            action_one_hot[ACTIONS.index(action)] = 1
        self.state_history.append(camera_data)
        self.action_history.append(action_one_hot)

    def get_state_sequence(self):
        state_sequence = list(self.state_history)
        action_sequence = list(self.action_history)
        while len(state_sequence) < SEQ_LENGTH:
            state_sequence.insert(0, np.zeros(5))  # Fixed buffer size for camera data
            action_sequence.insert(0, np.zeros(len(ACTIONS)))
        state_sequence = np.array(state_sequence).flatten()
        action_sequence = np.array(action_sequence).flatten()
        return np.concatenate((state_sequence, action_sequence))

    def is_move_possible(self, move):
        new_position = (self.position[0] + move[0], self.position[1] + move[1])
        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
            return True
        return False

class ExpertPolicy:
    def __init__(self, environment):
        self.env = environment

    def act(self, state, training=False):
        camera_data = self.env.get_camera_data()
        possible_moves = [move for move in ACTIONS if self.env.is_move_possible(move)]

        dog_detected = False
        dog_direction = None

        for i, val in enumerate(camera_data):
            if val == 2:
                move = (1, 0)
                if self.env.is_move_possible(move):
                    return move
            elif val == 3:
                dog_detected = True
                dog_direction = (i, 0)

        if dog_detected:
            possible_moves = [move for move in ACTIONS if self.env.is_move_possible(move)]
            if dog_direction in possible_moves:
                possible_moves.remove(dog_direction)
            if possible_moves:
                return random.choice(possible_moves)

        random.shuffle(possible_moves)
        for move in possible_moves:
            if self.env.is_move_possible(move):
                return move
        return random.choice(ACTIONS)

class NegativeExpertPolicy:
    def __init__(self, environment):
        self.env = environment

    def act(self, state, training=False):
        for dog_pos in self.env.dog_positions:
            move = (dog_pos[0] - self.env.position[0], dog_pos[1] - self.env.position[1])
            move = (np.sign(move[0]), np.sign(move[1]))
            if self.env.is_move_possible(move):
                return move
        return random.choice([move for move in ACTIONS if self.env.is_move_possible(move)])

class RandomPolicy:
    def __init__(self, environment):
        self.env = environment

    def act(self, state, training=False):
        return random.choice([move for move in ACTIONS if self.env.is_move_possible(move)])

class RepetitiveExpertPolicy:
    def __init__(self, environment, repetitive_action=(0, -1)):
        self.env = environment
        self.repetitive_action = repetitive_action

    def act(self, state, training=False):
        if self.env.is_move_possible(self.repetitive_action):
            return self.repetitive_action
        return random.choice([move for move in ACTIONS if self.env.is_move_possible(move)])

def generate_expert_data(env: GridEnvironment, num_episodes=1000, epsilon=0.1):
    print("Generating expert data...")
    positive_expert = ExpertPolicy(env)
    negative_expert = NegativeExpertPolicy(env)
    random_policy = RandomPolicy(env)
    repetitive_expert = RepetitiveExpertPolicy(env)
    data = []
    for episode in range(num_episodes):
        if episode % 2 == 0:
            state = env.reset(cats_count=1,dogs_count=3, training=True)
        else:
            state = env.reset(training=True)
        done = False
        move_count = 0
        while not done and move_count < 200:
            if random.random() < epsilon:
                action = random_policy.act(state, training=True)
            else:
                choice = random.random()
                if choice < 0.1:
                    action = negative_expert.act(state, training=True)
                elif 0.1 <= choice < 0.15:
                    action = repetitive_expert.act(state, training=True)
                else:
                    action = positive_expert.act(state, training=True)
            next_state, reward, done = env.step(action, training=True, cats_needed=1)
            if action in ACTIONS:
                action_index = ACTIONS.index(action)
                state_sequence = env.get_state_sequence()
                data.append((state_sequence, action_index, reward, next_state))
            state = next_state
            move_count += 1
    print("Expert data generation complete.")
    return data

class SimpleRNNRewardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(SimpleRNNRewardNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)
        c0 = torch.zeros(1, x.size(0), 64).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class IRLModel:
    def __init__(self, environment, expert_data, hidden_size=64):
        print("Initializing IRL model...")
        self.env = environment
        self.expert_data = expert_data
        input_size = 5 + len(ACTIONS)  # Fixed buffer size for camera data
        self.reward_model = SimpleRNNRewardNetwork(input_size, hidden_size, len(ACTIONS))
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        print("IRL model initialized")

    def train(self, num_iterations=1000, batch_size=16, second_phase=False, second_phase_iterations=0):
        print("Starting training...")
        self.losses = []
        for iteration in range(num_iterations):
            batch = random.sample(self.expert_data, batch_size)
            states, actions, rewards, next_states = zip(*batch)
            states = torch.tensor(np.array(states), dtype=torch.float32).reshape(batch_size, SEQ_LENGTH, -1)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            self.optimizer.zero_grad()
            outputs = self.reward_model(states)
            predicted_rewards = outputs.gather(1, actions.view(-1, 1)).squeeze()
            loss = self.criterion(predicted_rewards, rewards)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}")
        print("Training complete.")
        
        if second_phase:
            self.train_phase_two(second_phase_iterations)

    def train_phase_two(self, num_iterations=10000, batch_size=16):
        print("Starting phase two training...")
        self.losses_phase_two = []
        chunk_size = 10000
        num_chunks = num_iterations // chunk_size
        
        for chunk in range(num_chunks):
            print(f"Generating data for chunk {chunk + 1}/{num_chunks}")
            new_data = self.generate_data_using_policy(chunk_size)
            for iteration in range(chunk_size):
                batch = random.sample(new_data, batch_size)
                states, actions, rewards, next_states = zip(*batch)
                states = torch.tensor(np.array(states), dtype=torch.float32).reshape(batch_size, SEQ_LENGTH, -1)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                self.optimizer.zero_grad()
                outputs = self.reward_model(states)
                predicted_rewards = outputs.gather(1, actions.view(-1, 1)).squeeze()
                loss = self.criterion(predicted_rewards, rewards)
                loss.backward()
                self.optimizer.step()
                self.losses_phase_two.append(loss.item())
                if iteration % 100 == 0:
                    print(f"Phase Two - Chunk {chunk + 1}, Iteration {iteration}, Loss: {loss.item()}")
        print("Phase two training complete.")

    def generate_data_using_policy(self, num_episodes):
        print("Generating new data using learned policy...")
        policy = self.get_policy()
        new_data = []
        for episode in range(num_episodes):
            if episode % 100 == 0:
                print(f"Generating episode {episode + 1}/{num_episodes}.")
            state = self.env.reset(training=True)
            done = False
            move_count = 0
            while not done and move_count < 50:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, SEQ_LENGTH, -1)
                    action_values = policy(state_tensor)
                    sorted_actions = torch.argsort(action_values, descending=True)
                    for action_index in sorted_actions[0]:
                        action = ACTIONS[action_index.item()]
                        if self.env.is_move_possible(action):
                            break
                next_state, reward, done = self.env.step(action, training=True, cats_needed=1)
                if action in ACTIONS:
                    action_index = ACTIONS.index(action)
                    state_sequence = self.env.get_state_sequence()
                    new_data.append((state_sequence, action_index, reward, next_state))
                state = next_state
                move_count += 1
        print("New data generation complete.")
        return new_data


    def get_policy(self):
        return self.reward_model

def evaluate_policy(env: GridEnvironment, policy, num_episodes=10, cats_needed=1, grid_size=6, cats_count=2, dogs_count=2):
    print(f"Evaluating policy on a {grid_size}x{grid_size} grid with {cats_count} cats and {dogs_count} dogs...")
    video_filename = f'simulation_{grid_size}x{grid_size}_{cats_count}cats_{dogs_count}dogs_{cats_needed}needed.mp4'
    plot_filename = f'performance_plot_{grid_size}x{grid_size}_{cats_count}cats_{dogs_count}dogs_{cats_needed}needed.png'
    video_writer = create_video_writer(video_filename, 5, (640, 640))
    performance = []

    for episode in range(num_episodes):
        print(f"Evaluating Episode {episode + 1}/{num_episodes}")
        state = env.reset(cats_count=cats_count, dogs_count=dogs_count)
        done = False
        move_count = 0

        for attempt in range(30):
            # if not done:
            #     print(f"Attempt {attempt + 1}/30 for Episode {episode + 1}/{num_episodes}")
            while not done and move_count < 50:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, SEQ_LENGTH, -1)
                    action_values = policy(state_tensor)
                    sorted_actions = torch.argsort(action_values, descending=True)
                    for action_index in sorted_actions[0]:
                        action = ACTIONS[action_index.item()]
                        if env.is_move_possible(action):
                            break
                state, reward, done = env.step(action, cats_needed=cats_needed)
                add_frame_to_video(video_writer, env, episode)
                move_count += 1

            if not done:
                state = env.reset(training=False, preserve_positions=True, preserve_cats_found=True, cats_count=cats_count, dogs_count=dogs_count)
                move_count = 0

        performance.append(env.cats_reached - env.dogs_reached*dogs_count)
        print(f"Episode {episode + 1} completed, Performance: {performance[-1]}")

    video_writer.release()
    print("Evaluation complete. Saving performance plot...")

    plt.plot(performance)
    plt.xlabel('Episode')
    plt.ylabel('Performance (Cats reached - Dogs reached)')
    plt.title(f'Policy Performance Over Time on {grid_size}x{grid_size} grid')
    plt.savefig(plot_filename)

    video_writer = create_video_writer(f'simulation_final_{grid_size}x{grid_size}_{cats_count}cats_{dogs_count}dogs_{cats_needed}needed.mp4', 5, (640, 640))
    for _ in range(5):
        add_plot_to_video(video_writer, plot_filename)
    video_writer.release()
    print("Performance plot added to video.")

# for memory efficiency we want to generate the expert data 
# and use it to make the model then discard it
def make_model_from_expert_data(env, num_episodes):
    expert_data = generate_expert_data(env, num_episodes)

    irl_model = IRLModel(env, expert_data)
    return irl_model

# Update the main function to train and evaluate the model accordingly
def main():
    env = GridEnvironment(size=6)
    irl_model = make_model_from_expert_data(env, num_episodes=120000)

    print("IRL model created, about to start training")
    irl_model.train(num_iterations=120000, second_phase=True, second_phase_iterations=30000)

    learned_policy = irl_model.get_policy()
    evaluate_policy(env, learned_policy, num_episodes=10)  # Normal grid with 2 cats and 2 dogs

    env = GridEnvironment(size=6)  # Reset environment for next evaluation
    evaluate_policy(env, learned_policy, num_episodes=10, cats_needed=2)  # Normal grid but need 2 cats

    env = GridEnvironment(size=10)  # Larger grid with 3 cats and 3 dogs
    evaluate_policy(env, learned_policy, num_episodes=10, cats_needed=3, grid_size=10, cats_count=3, dogs_count=3)

if __name__ == "__main__":
    main()
