import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import sqlite3
from utils import load_and_resize_image, create_video_writer, add_frame_to_video, add_plot_to_video, display_environment

# Define the action space
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SEQ_LENGTH = 30  # Length of the action and sensor history

DATABASE_PATH = 'expert_data.db'
CACHE_SIZE = 10000  # Adjusted size to control memory usage

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
        self.visited_positions = set()
        self.reset(training=False)

    def reset(self, training=False, preserve_positions=False, cats_count=2, dogs_count=2, preserve_cats_found=False):
        if not preserve_positions:
            self.cat_positions = self.generate_positions(cats_count)
            self.dog_positions = self.generate_positions(dogs_count, avoid=self.cat_positions)
            
            all_occupied = self.cat_positions + self.dog_positions
            available_spaces = self.size * self.size - len(all_occupied)
            obstacle_count = min(dogs_count * 2 - 1, available_spaces - 4)
            
            self.obstacle_positions = self.generate_positions(obstacle_count, avoid=all_occupied) if obstacle_count > 0 else []
        
        if not preserve_cats_found:
            self.found_cats = [False] * len(self.cat_positions)
        
        self.moves = 0
        
        if not preserve_positions:
            all_occupied = set(self.cat_positions + self.dog_positions + self.obstacle_positions)
            available_positions = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in all_occupied]
            self.position = random.choice(available_positions) if available_positions else (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        
        self.cats_reached = sum(self.found_cats)
        self.dogs_reached = 0
        self.state_history = deque(maxlen=SEQ_LENGTH)
        self.action_history = deque(maxlen=SEQ_LENGTH)
        self.update_state()
        self.visited_positions = set()
        return self.get_state_sequence()

    def generate_positions(self, num_positions, min_distance=2, avoid=[]):
        positions = []
        attempts = 0
        max_attempts = num_positions * 100  # Limit total attempts

        while len(positions) < num_positions and attempts < max_attempts:
            position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if all(((position[0] - p[0])**2 + (position[1] - p[1])**2)**0.5 >= min_distance for p in positions + avoid):
                positions.append(position)
            attempts += 1

        return positions
    
    def get_camera_data(self):
        camera_view = np.zeros(5)  # Camera view for left, right, up, down, current
        x, y = self.position
        
        # Check all directions
        for i, direction in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            for j in range(1, self.size):
                check_x, check_y = x + j * direction[0], y + j * direction[1]
                if not (0 <= check_x < self.size and 0 <= check_y < self.size):
                    break
                if (check_x, check_y) in self.obstacle_positions:
                    break  # Stop if an obstacle is encountered
                if (check_x, check_y) in self.cat_positions:
                    camera_view[i] = 2
                    break
                elif (check_x, check_y) in self.dog_positions:
                    camera_view[i] = 3
                    break

        # Check current position
        if self.position in self.cat_positions:
            camera_view[4] = 2
        elif self.position in self.dog_positions:
            camera_view[4] = 3

        return camera_view

    def get_lidar_data(self):
        lidar_view = np.zeros(5)  # LIDAR view for left, right, up, down, current
        x, y = self.position
        
        # Check all directions
        for i, direction in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            for j in range(1, self.size):
                check_x, check_y = x + j * direction[0], y + j * direction[1]
                if not (0 <= check_x < self.size and 0 <= check_y < self.size):
                    break
                if (check_x, check_y) in self.obstacle_positions:
                    lidar_view[i] = 1 - (j - 1) / self.size  # Normalized distance to obstacle
                    break

        # Check current position
        if self.position in self.obstacle_positions:
            lidar_view[4] = 1

        return lidar_view

    def step(self, action, training=False, cats_needed=1):
        original_action = action
        new_position = (self.position[0] + action[0], self.position[1] + action[1])
        
        reward = -0.05  # Base step cost
        done = False
        
        if new_position in self.obstacle_positions:
            reward -= 0.1  # Penalty for trying to move into an obstacle
            possible_actions = [a for a in ACTIONS if (self.position[0] + a[0], self.position[1] + a[1]) not in self.obstacle_positions]
            if possible_actions:
                action = random.choice(possible_actions)
                new_position = (self.position[0] + action[0], self.position[1] + action[1])
            else:
                return self.get_state_sequence(), reward, done  # No valid move possible
        
        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
            self.position = new_position
            self.moves += 1
        else:
            reward -= 0.05  # Small additional cost for attempting an impossible move

        # Simplified reward calculation
        camera_data = self.get_camera_data()
        if 2 in camera_data[:4]:
            reward += 0.1  # Reward for being near a cat
        if 3 in camera_data[:4]:
            reward -= 0.1  # Penalty for being near a dog

        if self.position not in self.visited_positions:
            reward += 0.1  # Reward for visiting a new position
            self.visited_positions.add(self.position)

        if self.position in self.cat_positions:
            idx = self.cat_positions.index(self.position)
            if not self.found_cats[idx]:
                self.found_cats[idx] = True
                reward += 1.0
                self.cats_reached += 1

        if self.position in self.dog_positions:
            reward -= 5.0
            self.dogs_reached += 1
            done = True

        if self.cats_reached >= cats_needed:
            done = True
            reward += 10.0

        if self.moves >= 200:
            done = True

        self.record_state(original_action)
        return self.get_state_sequence(), reward, done

    def update_state(self):
        camera_data = self.get_camera_data()
        lidar_data = self.get_lidar_data()
        self.state_history.append(np.concatenate([camera_data, lidar_data]))
        self.action_history.append(np.zeros(len(ACTIONS)))

    def record_state(self, action):
        camera_data = self.get_camera_data()
        lidar_data = self.get_lidar_data()
        action_one_hot = np.zeros(len(ACTIONS))
        if action in ACTIONS:
            action_one_hot[ACTIONS.index(action)] = 1
        self.state_history.append(np.concatenate([camera_data, lidar_data]))
        self.action_history.append(action_one_hot)

    def get_state_sequence(self):
        state_sequence = list(self.state_history)
        action_sequence = list(self.action_history)
        while len(state_sequence) < SEQ_LENGTH:
            state_sequence.insert(0, np.zeros(10))  # 5 for camera, 5 for LIDAR
            action_sequence.insert(0, np.zeros(len(ACTIONS)))
        state_sequence = np.array(state_sequence).flatten()
        action_sequence = np.array(action_sequence).flatten()
        return np.concatenate((state_sequence, action_sequence))

    def is_move_possible(self, move):
        new_position = (self.position[0] + move[0], self.position[1] + move[1])
        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
            return True
        return False

class AntiRepetitionMixin:
    def __init__(self, max_repetitions=2):
        self.recent_moves = deque(maxlen=max_repetitions)
        self.max_repetitions = max_repetitions

    def avoid_repetition(self, possible_moves):
        if len(self.recent_moves) == self.max_repetitions and len(set(self.recent_moves)) == 1:
            non_repetitive_moves = [move for move in possible_moves if move != self.recent_moves[-1]]
            return non_repetitive_moves if non_repetitive_moves else possible_moves
        return possible_moves

    def update_recent_moves(self, move):
        self.recent_moves.append(move)

class ExpertPolicy(AntiRepetitionMixin):
    def __init__(self, environment, max_repetitions=2):
        super().__init__(max_repetitions)
        self.env = environment

    def act(self, state, training=False):
        camera_data = self.env.get_camera_data()
        lidar_data = self.env.get_lidar_data()
        possible_moves = [move for move in ACTIONS if self.env.is_move_possible(move)]
        possible_moves = self.avoid_repetition(possible_moves)

        # If a dog is visible, prioritize moving away
        for i, val in enumerate(camera_data[:4]):
            if val == 3:  # Dog detected
                opposite_move = ACTIONS[(i + 2) % 4]  # Move in opposite direction
                if opposite_move in possible_moves:
                    self.update_recent_moves(opposite_move)
                    return opposite_move

        # If a cat is visible, move towards it
        for i, val in enumerate(camera_data[:4]):
            if val == 2:  # Cat detected
                move = ACTIONS[i]
                if move in possible_moves:
                    self.update_recent_moves(move)
                    return move

        # If no cats or dogs visible, explore while avoiding obstacles
        safe_moves = [move for move in possible_moves if lidar_data[ACTIONS.index(move)] == 0]
        if safe_moves:
            move = random.choice(safe_moves)
            self.update_recent_moves(move)
            return move

        # If cornered, make any possible move
        move = random.choice(possible_moves) if possible_moves else random.choice(ACTIONS)
        self.update_recent_moves(move)
        return move

class RealisticExpertPolicy(AntiRepetitionMixin):
    def __init__(self, environment, error_rate=0.2, memory_length=5, max_repetitions=2):
        super().__init__(max_repetitions)
        self.env = environment
        self.error_rate = error_rate
        self.memory_length = memory_length
        self.cat_memory = deque(maxlen=memory_length)

    def act(self, state, training=False):
        camera_data = self.env.get_camera_data()
        lidar_data = self.env.get_lidar_data()
        possible_moves = [move for move in ACTIONS if self.env.is_move_possible(move)]
        possible_moves = self.avoid_repetition(possible_moves)

        # Update cat memory
        cat_positions = [i for i, val in enumerate(camera_data[:4]) if val == 2]
        if cat_positions:
            self.cat_memory.append(cat_positions[0])
        elif len(self.cat_memory) > 0:
            self.cat_memory.append(self.cat_memory[-1])

        # If a dog is visible, prioritize moving away
        for i, val in enumerate(camera_data[:4]):
            if val == 3:  # Dog detected
                opposite_move = ACTIONS[(i + 2) % 4]  # Move in opposite direction
                if opposite_move in possible_moves:
                    return self.apply_error(opposite_move, possible_moves)

        # If a cat is visible, move towards it
        for i, val in enumerate(camera_data[:4]):
            if val == 2:  # Cat detected
                move = ACTIONS[i]
                if move in possible_moves:
                    return self.apply_error(move, possible_moves)

        # If cat was recently seen, move towards its last known position
        if self.cat_memory:
            last_known_cat_direction = self.cat_memory[-1]
            move = ACTIONS[last_known_cat_direction]
            if move in possible_moves:
                return self.apply_error(move, possible_moves)

        # If no cats or dogs visible, explore while avoiding obstacles
        safe_moves = [move for move in possible_moves if lidar_data[ACTIONS.index(move)] == 0]
        if safe_moves:
            return self.apply_error(random.choice(safe_moves), possible_moves)

        # If cornered, make any possible move
        return self.apply_error(random.choice(possible_moves) if possible_moves else random.choice(ACTIONS), possible_moves)

    def apply_error(self, intended_move, possible_moves):
        if random.random() < self.error_rate:
            wrong_moves = [move for move in possible_moves if move != intended_move]
            move = random.choice(wrong_moves) if wrong_moves else intended_move
        else:
            move = intended_move
        self.update_recent_moves(move)
        return move

class NegativeExpertPolicy(AntiRepetitionMixin):
    def __init__(self, environment, max_repetitions=2):
        super().__init__(max_repetitions)
        self.env = environment

    def act(self, state, training=False):
        camera_data = self.env.get_camera_data()
        possible_moves = [move for move in ACTIONS if self.env.is_move_possible(move) and 
                          (self.env.position[0] + move[0], self.env.position[1] + move[1]) not in self.env.obstacle_positions]
        possible_moves = self.avoid_repetition(possible_moves)
        
        # If a dog is visible, move towards it
        for i, val in enumerate(camera_data[:4]):
            if val == 3:  # Dog detected
                move = ACTIONS[i]
                if move in possible_moves:
                    self.update_recent_moves(move)
                    return move
        
        move = random.choice(possible_moves) if possible_moves else random.choice(ACTIONS)
        self.update_recent_moves(move)
        return move

class RandomPolicy(AntiRepetitionMixin):
    def __init__(self, environment, max_repetitions=2):
        super().__init__(max_repetitions)
        self.env = environment

    def act(self, state, training=False):
        possible_moves = [move for move in ACTIONS if self.env.is_move_possible(move) and 
                          (self.env.position[0] + move[0], self.env.position[1] + move[1]) not in self.env.obstacle_positions]
        possible_moves = self.avoid_repetition(possible_moves)
        move = random.choice(possible_moves) if possible_moves else random.choice(ACTIONS)
        self.update_recent_moves(move)
        return move

def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expert_data (
                id INTEGER PRIMARY KEY,
                state_sequence TEXT,
                action_index INTEGER,
                reward REAL,
                next_state TEXT
            )
        ''')
        conn.commit()

def generate_expert_data(env: GridEnvironment, num_sequences=1000, load_previous=False):
    print("Generating expert data...")
    realistic_expert = RealisticExpertPolicy(env)
    perfect_expert = ExpertPolicy(env)
    negative_expert = NegativeExpertPolicy(env)
    random_policy = RandomPolicy(env)
    
    initialize_database()
    
    # Check how many sequences are already in the database
    existing_sequences = 0
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM expert_data')
        existing_sequences = cursor.fetchone()[0]
    
    # Calculate how many sequences need to be generated
    sequences_to_generate = num_sequences - existing_sequences
    if sequences_to_generate > 0:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            for episode in range(sequences_to_generate):
                if episode % 10000 == 0:
                    print(f"{episode}")
                scenario = random.choices(
                    [
                        {"cats": 1, "dogs": 1, "weight": 0.1},
                        {"cats": 2, "dogs": 1, "weight": 0.3},
                        {"cats": 2, "dogs": 2, "weight": 0.2},
                        {"cats": 2, "dogs": 3, "weight": 0.2},
                        {"cats": 3, "dogs": 3, "weight": 0.2},
                    ], 
                    weights=[0.1, 0.3, 0.2, 0.2, 0.2]
                )[0]
                
                while True:
                    state = env.reset(cats_count=scenario["cats"], dogs_count=scenario["dogs"], training=True)
                    done = False
                    move_count = 0
                    episode_data = []

                    while not done and move_count < 200:
                        policy_choice = random.random()
                        if policy_choice < 0.5:
                            action = realistic_expert.act(state, training=True)
                        elif policy_choice < 0.7:
                            action = perfect_expert.act(state, training=True)
                        elif policy_choice < 0.9:
                            action = negative_expert.act(state, training=True)
                        else:
                            action = random_policy.act(state, training=True)

                        next_state, reward, done = env.step(action, training=True, cats_needed=scenario["cats"])
                        if action in ACTIONS:
                            action_index = ACTIONS.index(action)
                            state_sequence = env.get_state_sequence().tolist()
                            next_state_sequence = next_state.tolist()
                            episode_data.append((str(state_sequence), action_index, reward, str(next_state_sequence)))
                        state = next_state
                        move_count += 1

                    if env.cats_reached >= scenario["cats"]:
                        break  # Ensure 100% success for the expert policies

                for data in episode_data:
                    cursor.execute('''
                        INSERT INTO expert_data (state_sequence, action_index, reward, next_state)
                        VALUES (?, ?, ?, ?)
                    ''', data)
            conn.commit()
    
    print("Expert data generation complete.")

def load_data_for_training(batch_size, cache_size=CACHE_SIZE):
    print("Loading data for training...")
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM expert_data')
        total_lines = cursor.fetchone()[0]
        print(f"Total lines in database: {total_lines}")

        cache = []
        while True:
            if len(cache) < batch_size:
                cursor.execute('SELECT * FROM expert_data ORDER BY RANDOM() LIMIT ?', (cache_size,))
                cache.extend(cursor.fetchall())
                print(f"Cache refilled with {len(cache)} records.")
            batch = cache[:batch_size]
            cache = cache[batch_size:]
            yield batch

class EfficientAttention(nn.Module):
    def __init__(self, hidden_size):
        super(EfficientAttention, self).__init__()
        self.scale = 1.0 / (hidden_size ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = self.softmax(scores)
        return torch.matmul(weights, value)

class SimpleRNNRewardNetworkWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(SimpleRNNRewardNetworkWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = EfficientAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)
        c0 = torch.zeros(1, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.attention(out, out, out)  # Self-attention
        out = self.fc(out[:, -1, :])
        return out

class IRLModel:
    def __init__(self, environment, hidden_size=64):
        print("Initializing IRL model with attention...")
        self.env = environment
        input_size = 10 + len(ACTIONS)  # 5 for camera, 5 for LIDAR, and actions
        self.reward_model = SimpleRNNRewardNetworkWithAttention(input_size, hidden_size, len(ACTIONS))
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        print("IRL model with attention initialized")

    def train(self, num_iterations=1000, batch_size=16, second_phase=False, second_phase_iterations=0):
        print("Starting training...")
        self.losses = []
        batch_generator = load_data_for_training(batch_size)
        for iteration in range(num_iterations):
            try:
                batch = next(batch_generator)
                states, actions, rewards, next_states = zip(*[self.parse_row(row) for row in batch])
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
                    print(f"Processing batch of size {len(batch)}")
                    print(f"Iteration {iteration}, Loss: {loss.item()}")
            except StopIteration:
                print("No more data to process.")
                break
            except Exception as e:
                print(f"Error during training at iteration {iteration}: {e}")
        print("Training complete.")
        
        if second_phase:
            self.train_phase_two(second_phase_iterations)

    def parse_row(self, row):
        _, state_sequence, action_index, reward, next_state = row
        state_sequence = np.array(eval(state_sequence))
        next_state = np.array(eval(next_state))
        return state_sequence, int(action_index), float(reward), next_state

    def train_phase_two(self, num_iterations=1000, batch_size=16):
        print("Starting phase two training...")
        self.losses_phase_two = []
        successful_episodes = 0
        new_data = []

        while successful_episodes < num_iterations:
            episode_data = self.generate_data_using_policy(1)  # Generate one episode at a time
            if episode_data:  # If episode was successful (not rejected)
                new_data.extend(episode_data)

            if len(new_data) >= batch_size:
                batch = new_data[:batch_size]
                new_data = new_data[batch_size:]
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
                successful_episodes += 1
                if successful_episodes % 1000 == 0:
                    print(f"Phase Two - Successful Episode {successful_episodes}, Loss: {loss.item()}")
        print("Phase two training complete.")

    def generate_data_using_policy(self, num_episodes):
        print("Generating new data using learned policy...")
        policy = self.get_policy()
        new_data = []
        for episode in range(num_episodes):
            if episode % 1000 == 0:
                print(f"Generating episode {episode + 1}/{num_episodes}.")
            state = self.env.reset(training=True, cats_count=3, dogs_count=3)
            done = False
            move_count = 0
            episode_data = []
            while not done and move_count < 50:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, SEQ_LENGTH, -1)
                    action_values = policy(state_tensor)
                    sorted_actions = torch.argsort(action_values, descending=True)
                    for action_index in sorted_actions[0]:
                        action = ACTIONS[action_index.item()]
                        if self.env.is_move_possible(action):
                            break
                next_state, reward, done = self.env.step(action, training=True, cats_needed=3)
                if action in ACTIONS:
                    action_index = ACTIONS.index(action)
                    state_sequence = self.env.get_state_sequence()
                    episode_data.append((state_sequence, action_index, reward, next_state))
                state = next_state
                move_count += 1

            # Reject 70% of the episodes where the objective is not achieved
            if self.env.cats_reached < 3 and random.random() < 0.7:
                continue

            new_data.extend(episode_data)

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

        performance.append(env.cats_reached - env.dogs_reached * dogs_count)
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

def make_model_from_expert_data(env, num_sequences, load_previous = False):
    generate_expert_data(env, num_sequences, load_previous)
    irl_model = IRLModel(env)
    return irl_model

def main():
    env = GridEnvironment(size=10)
    irl_model = make_model_from_expert_data(env, num_sequences=200000, load_previous=False)

    print("IRL model created, about to start training")
    irl_model.train(num_iterations=100000, batch_size=16, second_phase=True, second_phase_iterations=50000)

    learned_policy = irl_model.get_policy()

    # Save the trained model
    torch.save(learned_policy.state_dict(), 'latest_policy.pth')
    print("Trained model saved as 'latest_policy.pth'")

    evaluate_policy(env, learned_policy, num_episodes=10)  # Normal grid with 2 cats and 2 dogs

    env = GridEnvironment(size=6)  # Reset environment for next evaluation
    evaluate_policy(env, learned_policy, num_episodes=10, cats_needed=2)  # Normal grid but need 2 cats

    env = GridEnvironment(size=10)  # Larger grid with 3 cats and 3 dogs
    evaluate_policy(env, learned_policy, num_episodes=10, cats_needed=3, grid_size=10, cats_count=3, dogs_count=3)

if __name__ == "__main__":
    main()
