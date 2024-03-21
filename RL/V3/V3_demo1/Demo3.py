"""
The goal of this script is to implement the training for a neural network implementing the robot navigation for the minipupper 2 robot.
This script utilises a grid-based environment simulation to train the robot to target cats and avoid dogs and obstacles whilst navigating in ideally the most efficient way.
It aims to find teach the robot to navigate and efficiently scan/patrol the navigatable environment.

This training regime combines several machine learning methodsL behavioral cloning, reinforcement learning (RL), and curriculum learning.

### ML methods overview
1. **Behavioural Cloning**: Behavioral cloning is a method of generating a corpus of expert training data. The robots neural network is then taught to mimik the expert strategies.
2. **RL**: This is implemented in the second training phase section, RL is where the robot utilises its NN to select the next move. Each time it moves it is punished for a bad action and rewarded for a good action. The NN then learns based on this reward function and improves itself through attempting the environment and getting these rewards and punishments.
3. **Curriculum Learning** Curriculum learning is where the robot is exposed to different tasks of increasing complexity. Helping the robot to learn through a more logical learning curve.

### Neural Networks
1. **RecurrentNeuralNetworkRewardNetwork**: A recurrent neural network that predicts rewards based on sequences of states and actions. It captures long-term dependencies in the action sequences, enabling the robot to make informed decisions.
2. **ShortTermContextNetwork**: A fully connected neural network that makes decisions based on immediate sensor data and actions. It helps the robot avoid immediate dangers and make quick decisions.
3. **DualNetworkPolicy**: Integrates the RecurrentNeuralNetworkRewardNetwork and ShortTermContextNetwork to make decisions considering both long-term and short-term contexts.


### Databases
This code uses SQLite databases to store and generate data such that it doesnt need to all be held in RAM
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import sqlite3
import multiprocessing as mp
import pygame
import time
import sqlite3
import os
import sqlite3
import traceback
import sys
from io import StringIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# you will need to use the following to pip install all requirements
# pip install numpy torch matplotlib pygame Pillow
# please add anything left out to this list

import utils # from the utils file

# Suppress the annoying pygame messages
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
stdout = sys.stdout
sys.stdout = StringIO()
sys.stdout = stdout
from PIL import Image, ImageDraw

# Define the possible actions
ACTIONS = ['turn_left', 'turn_right', 'move_forward']
# History length, at each history position, the move taken and all detections are stored
SEQUENCE_LENGTH = 15  # so there will be 15 times the camera detection possibilities + 4 lidar detection possibilities + the move taken (which is a size of 3 because we need false/0 for the action not taken)

DB_PATH = 'data.db'  # SQL database is used to minimise RAM usage at one time
DB_CACHE_SIZE = 10000  # Adjusted size to control memory usage

# Manage the grid for the simulation
class SimulationGridEnvironment:
    def __init__(self, size=6, cats_count=2, dogs_count=2):
        """
        Grid environemnt initialisation.
        Parameters:
        - grid_size: grid_size x grid_size -> size of grid
        - num_cats: Number of cats (target positions) in the grid
        - num_dogs: Number of dogs (sources of dangers) in the grid
        """
        self.size = size
        self.cell_height, self.cell_width = 64, 64  # cell size (useful for the visualisation)
        self.grid_height = self.cell_height * self.size
        self.grid_width = self.cell_width * self.size

        # We want to be able to track positions and angles visited during training, we can't do this in the real world on the robot but its fine for helping with training and aids with giving reward funciton options.
        self.visited_states = set()
        self.reset(cats_count=cats_count, dogs_count=dogs_count)

    def reset(self, preserve_positions=False, return_old_self_position=False, cats_count=2, dogs_count=2, preserve_cats_found=False, custom_positions=None, train_avoidance_network_mode=False):
        """
        Reset the environment.
        The environment can be reset in a variety of ways.

        Parameters:
        - preserve_positions:  previous positions should be preserved
        - return_old_position: robot should return to its old position
        - num_cats: Number of cats in grid (no effect for preserve positions)
        - num_dogs: Number of dogs in grid (no effect for preserve positions)
        - preserve_cats_found: previously found cats should be preserved
        - custom_positions: USe custom positions for initializing the grid instead or random initialisation
        - short_term_context_mode: indicates this is for training the short-term context network and thus needs custom reset
        """
        if custom_positions:
            self.cat_positions = custom_positions.get('cat_positions', self.position_generation(cats_count))
            self.dog_positions = custom_positions.get('dog_positions', self.position_generation(dogs_count, avoid=self.cat_positions))
            self.obstacle_positions = custom_positions.get('obstacle_positions', self.position_generation(dogs_count * 2 - 1, avoid=self.cat_positions + self.dog_positions))
            self.position = custom_positions.get('robot_position', self.safe_position_generation())
            self.direction = custom_positions.get('robot_direction', random.randint(0, 3))

        else:
            if return_old_self_position:
                self.position = self.previous_reset_position
                self.direction = self.previous_reset_direction

            if not preserve_positions:
                self.cat_positions = self.position_generation(cats_count)
                self.dog_positions = self.position_generation(dogs_count, avoid=self.cat_positions)

                all_occupied = self.cat_positions + self.dog_positions
                available_spaces = self.size * self.size - len(all_occupied)

                if train_avoidance_network_mode == False:
                    obstacle_count = min(dogs_count * 3 - 1, available_spaces - 4)

                    self.obstacle_positions = self.position_generation(obstacle_count, avoid=all_occupied) if obstacle_count > 0 else []

            if not preserve_cats_found:
                self.found_cats = [False] * len(self.cat_positions)

            self.moves = 0

            if not preserve_positions:
                if not return_old_self_position:
                    if train_avoidance_network_mode == True:
                         position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                    else:
                        self.position = self.safe_position_generation()
                    self.direction = random.randint(0, 3)  # Random initial direction
                    self.previous_reset_position = self.position
                    self.previous_reset_direction = self.direction

        # Ensure self.found_cats has the correct length
        if not preserve_cats_found or len(self.found_cats) != len(self.cat_positions):
            self.found_cats = [False] * len(self.cat_positions)

        self.cats_reached = sum(self.found_cats)
        self.dogs_reached = 0
        self.state_history = deque(maxlen=SEQUENCE_LENGTH)
        self.action_history = deque(maxlen=SEQUENCE_LENGTH)
        self.visited_states = set()
        self.update_state()
        return self.get_memory_state_sequence()

    def safe_position_generation(self):
        """
        create a safe starting position for the robot where it isnt cornered and useless
        """
        while True:
            position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.check_if_position_is_viable(position):
                return position

    def check_if_position_is_viable(self, position):
        """
        check position isn't cornered and useless
        """
        for dog_pos in self.dog_positions:
            if self.manhattan_distance(position, dog_pos) <= 3:
                return False
        return position not in self.cat_positions + self.dog_positions + self.obstacle_positions

    @staticmethod
    def manhattan_distance(pos1, pos2):
        """
        use manhattan distance instead of euler because its a better count of grid spaces between the robot and obstacels
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def position_generation(self, num_positions, min_distance=2, avoid=[]):
        """
        generate viable positions that arent too close to be useless
        """
        positions = []
        attempts = 0
        max_attempts = num_positions * 100  # don't allow unlimited attempts

        # positions randomly generated but checked for viability
        while len(positions) < num_positions and attempts < max_attempts:
            position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if all(((position[0] - p[0])**2 + (position[1] - p[1])**2)**0.5 >= min_distance for p in positions + avoid):
                positions.append(position)
            attempts += 1

        return positions
    
    def get_sensor_data(self, sensor_type='camera'):
        """
        Get data from the sensors
        """
        # TODO: no reason for 4 directions for camera if we only care about infront
        # could be a big efficincy upgrade to fix this, I think N, E, S, W is no longer the most relevant,
        # even for lidar we can just have infront left right and behind
        # camera currently does only check infront so definitely no reason to have all these directions for camera



        view = np.zeros(4)  # [front, right, back, left]
        x, y = self.position
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W

        
        for i in range(4):
            check_direction = (self.direction + i) % 4
            dx, dy = directions[check_direction]
            
            for j in range(1, self.size):
                check_x, check_y = x + j * dx, y + j * dy
                if not (0 <= check_x < self.size and 0 <= check_y < self.size):
                    break
                if (check_x, check_y) in self.obstacle_positions:
                    if sensor_type == 'lidar':
                        view[i] = 1 - (j - 1) / self.size  # Normalized distance 
                    break
                if sensor_type == 'camera' and i == 0:  # Only check for cats/dogs in front
                    if (check_x, check_y) in self.cat_positions:
                        view[i] = 2
                        break
                    elif (check_x, check_y) in self.dog_positions:
                        if j <= 3: # don't worry about far away dogs
                            view[i] = 3 
                        break
        
        return view

    def step_reward_function(self, action, training=False, cats_needed=1):
        """
        Perform an action and train based on this action.
        Reward function defined here
        Parameters:
        - action: Action the robot will take
        - training: Will trainging take place?
        - cats_needed: Cats needed to return done
        """
        reward = -0.01  # Base step cost
        done = False
        
        old_position = self.position
        old_direction = self.direction

        if action == 'turn_left':
            self.direction = (self.direction - 1) % 4
        elif action == 'turn_right':
            self.direction = (self.direction + 1) % 4
        elif action == 'move_forward':
            direction_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            dx, dy = direction_deltas[self.direction]
            new_position = (self.position[0] + dx, self.position[1] + dy)
            
            if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
                if new_position not in self.obstacle_positions:
                    self.position = new_position
                    self.moves += 1
                else:
                    reward -= 0.1  # Penalty for trying to move into an obstacle
            else:
                reward -= 0.05  # Small cost for attempting an impossible move

        # Check if this position-direction pair is new
        current_state = (*self.position, self.direction)
        if current_state not in self.visited_states:
            reward += 0.1  # Reward for visiting a new position-direction pair
            self.visited_states.add(current_state)

        camera_data = self.get_sensor_data('camera')
        if camera_data[0] == 2:
            reward += 0.1  # Reward for being near a cat
        elif camera_data[0] == 3:
            reward -= 0.1  # Penalty for being near a dog

        if self.position in self.cat_positions:
            idx = self.cat_positions.index(self.position)
            if idx < len(self.found_cats) and not self.found_cats[idx]:
                self.found_cats[idx] = True
                reward += 1.0
                self.cats_reached += 1

        # worse thing the robot can do is be where the dog is, punish it bigly
        if self.position in self.dog_positions:
            reward -= 5.0
            self.dogs_reached += 1
            done = True

        if self.cats_reached >= cats_needed:
            done = True
            # we want to give a bonus reward here if there is only 1 or 2 cats present during training
            # so the robot doesnt think it did worse simply because there are less possible cats
            reward += 10.0 + 3 - cats_needed  # assuming max cats during training is 3

        if self.moves >= 500:
            done = True

        self.record_state(action)
        return self.get_memory_state_sequence(), reward, done
    
    def update_state(self):
        """
        update the camera and lidar data and the state and action history
        Set the robots current direction to 1
        """
        camera_data = self.get_sensor_data('camera')
        lidar_data = self.get_sensor_data('lidar')
        direction_one_hot = [0, 0, 0, 0]
        direction_one_hot[self.direction] = 1
        self.state_history.append(np.concatenate([camera_data, lidar_data, direction_one_hot]))
        self.action_history.append(np.zeros(len(ACTIONS)))

    def record_state(self, action):
        """
        Add state to history
        """
        camera_data = self.get_sensor_data('camera')
        lidar_data = self.get_sensor_data('lidar')
        direction_one_hot = [0, 0, 0, 0]
        direction_one_hot[self.direction] = 1
        state = np.concatenate([camera_data, lidar_data, direction_one_hot])
        
        action_one_hot = np.zeros(len(ACTIONS))
        if action in ACTIONS:
            action_one_hot[ACTIONS.index(action)] = 1
        
        self.state_history.append(state)
        self.action_history.append(action_one_hot)

    def get_memory_state_sequence(self):
        """
        get the memory sequence of past states
        """
        state_sequence = list(self.state_history)
        action_sequence = list(self.action_history)
        
        # Ensure all state elements have the same length (12 in this case: 4 for camera, 4 for LIDAR, 4 for direction)
        state_length = 12
        
        # Pad the sequences if they're not full
        while len(state_sequence) < SEQUENCE_LENGTH:
            state_sequence.insert(0, np.zeros(state_length))
            action_sequence.insert(0, np.zeros(len(ACTIONS)))
        
        # Ensure all elements in state_sequence have the same length
        state_sequence = [np.array(s).flatten()[:state_length] for s in state_sequence]
        
        state_sequence = np.array(state_sequence).flatten()
        action_sequence = np.array(action_sequence).flatten()
        
        return np.concatenate((state_sequence, action_sequence))

    def check_if_move_possible(self, action):
        """
        Check the move doesnt exit the borders
        Check move doesnt overlap obstacle
        This does not allow for dog avoidance
        That must be done by the NN not here
        """
        if action in ['turn_left', 'turn_right']:
            return True
        elif action == 'move_forward':
            # Define direction deltas: 0: North, 1: East, 2: South, 3: West
            direction_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            dx, dy = direction_deltas[self.direction]
            new_position = (self.position[0] + dx, self.position[1] + dy)
            return (0 <= new_position[0] < self.size and 
                    0 <= new_position[1] < self.size and 
                    new_position not in self.obstacle_positions)
        return False

# Extends the base grid env to allow for pygame visualisation
class SimulationGridVisualisationEnvironment(SimulationGridEnvironment):
    def __init__(self, size=6, dog_image_path='dog.png', cat_image_path='cat.png', robot_image_path='robot.png'):
        super().__init__(size)
        """
        Grid environment inherited class for loading pygame visualisation
        """
        self.dog_image_path = dog_image_path
        self.cat_image_path = cat_image_path
        self.robot_image_path = robot_image_path
        self.cell_height, self.cell_width = 64, 64  
        self.grid_height = self.cell_height * self.size
        self.grid_width = self.cell_width * self.size
        self.pygame_initialized = False
        self.direction = 0 

        self.dog_image, self.cat_image, self.robot_image = utils.initialize_pygame(dog_image_path, cat_image_path, robot_image_path)

    def get_frame(self):
        return utils.get_frame(self, self.dog_image, self.cat_image, self.robot_image)

class SQLiteDatabaseHandler:
    def __init__(self, db_path):
        self.db_path = db_path

    def initialize_db(self):
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            with sqlite3.connect(self.db_path, timeout=20) as conn:
                cursor = conn.cursor()
                
                start_time = time.time()
                timeout = 30  # 30 seconds timeout
                
                while True:
                    try:
                        cursor.execute('''
                            CREATE TABLE IF NOT EXISTS expert_data (
                                id INTEGER PRIMARY KEY,
                                state_sequence TEXT,
                                action_index INTEGER,
                                reward REAL,
                                next_state TEXT
                            )
                        ''')
                        break  # If successful, exit  loop
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e):
                            if time.time() - start_time > timeout:
                                raise TimeoutError("Table creation timed out after 30 seconds")
                            print("Database locked. Retry in 1 second...")
                            time.sleep(1)
                        else:
                            raise  # Re-raise exception if it's not a "database is locked" error
                
                conn.commit()
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(f"An SQLite error occurred: {e}")
            print("SQLite traceback:")
            traceback.print_exc()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("traceback:")
            traceback.print_exc()

    def clear_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM expert_data')
                conn.commit()
            print("Database cleared successfully")
        except Exception as e:
            print(f"An error occurred while clearing the database: {e}")
            traceback.print_exc()

    def load_data_for_training(self, batch_size, cache_size=DB_CACHE_SIZE):
        """
        loads expert data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM expert_data')
            total_lines = cursor.fetchone()[0]

            cache = []
            while True:
                if len(cache) < batch_size:
                    cursor.execute('SELECT * FROM expert_data ORDER BY RANDOM() LIMIT ?', (cache_size,))
                    cache.extend(cursor.fetchall())
                batch = cache[:batch_size]
                cache = cache[batch_size:]
                yield batch

class AntiRepetitionMixin:
    """
    Prevent repetition in expert data
    """
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
    def __init__(self, environment: SimulationGridEnvironment, max_repetitions=2):
        """
        initialise an expert policy
        """
        super().__init__(max_repetitions)
        self.env = environment

    def act(self, state, training=False):
        raise NotImplementedError("Policy classes must implement the act method")

class RealisticExpertPolicy(ExpertPolicy):
    """
    realistic expert acting in error prone environment (sensor errors exist in real world)
    practicing recovery from unexpected changes
    """
    def __init__(self, environment: SimulationGridEnvironment, error_rate=0.2, memory_length=5, max_repetitions=2):
        super().__init__(environment, max_repetitions)
        self.error_rate = error_rate
        self.memory_length = memory_length
        self.target_memory = deque(maxlen=memory_length)

    def act(self, state, training=False):
        camera_data = self.env.get_sensor_data('camera')
        lidar_data = self.env.get_sensor_data('lidar')
        possible_moves = [move for move in ACTIONS if self.env.check_if_move_possible(move)]
        possible_moves = self.avoid_repetition(possible_moves)

        # Update memory
        if camera_data[0] in [2, 3]:
            self.target_memory.append(camera_data[0])
        elif len(self.target_memory) > 0:
            self.target_memory.append(self.target_memory[-1])

        # If no target is visible, look around
        if camera_data[0] == 0:
            return self.apply_error(random.choice(['turn_left', 'turn_right']), possible_moves)

        # Check what's directly in front
        if camera_data[0] == 2:  # Cat in front
            return self.apply_error('move_forward', possible_moves)
        elif camera_data[0] == 3:  # Dog in front
            return self.apply_error(random.choice(['turn_left', 'turn_right']), possible_moves)

        # If no immediate threats or goals, explore based on memory or randomly
        if self.target_memory:
            last_known_target = self.target_memory[-1]
            if last_known_target == 2:  # Last known target was a cat
                return self.apply_error('move_forward', possible_moves)
            else:  # Last known target was a dog
                return self.apply_error(random.choice(['turn_left', 'turn_right']), possible_moves)

        if 'move_forward' in possible_moves and lidar_data[0] == 0:
            return self.apply_error('move_forward', possible_moves)
        else:
            return self.apply_error(random.choice(['turn_left', 'turn_right']), possible_moves)

    def apply_error(self, intended_move, possible_moves):
        if random.random() < self.error_rate:
            wrong_moves = [move for move in possible_moves if move != intended_move]
            move = random.choice(wrong_moves) if wrong_moves else intended_move
        else:
            move = intended_move
        self.update_recent_moves(move)
        return move

class ExpertPolicy(ExpertPolicy):
    """
    The expert should always go towards a cat and away from a dog
    train this behaviour for behavioural cloning
    """
    def act(self, state, training=False):
        camera_data = self.env.get_sensor_data('camera')
        lidar_data = self.env.get_sensor_data('lidar')
        possible_moves = [move for move in ACTIONS if self.env.check_if_move_possible(move)]
        possible_moves = self.avoid_repetition(possible_moves)

        # Check what's directly in front
        if camera_data[0] == 2:  # Cat in front
            self.update_recent_moves('move_forward')
            return 'move_forward'
        elif camera_data[0] == 3:  # Dog in front
            turn = random.choice(['turn_left', 'turn_right'])
            self.update_recent_moves(turn)
            return turn

        # If no immediate threats or goals, explore
        if 'move_forward' in possible_moves and lidar_data[0] == 0:
            # looking around is good but don't want to do all the time so weight towards move forward
            move = random.choice(['turn_left', 'turn_right', 'move_forward', 'move_forward', 'move_forward', 'move_forward'])
            self.update_recent_moves(move)
            return 'move_forward'
        else:
            turn = random.choice(['turn_left', 'turn_right'])
            self.update_recent_moves(turn)
            return turn

        
class NegativeExpertPolicy(ExpertPolicy):
    """
    We want some examples of what not to do in the behavioural cloning
    purely so that the negative rewards are known in some way
    """
    def act(self, state, training=False):
        camera_data = self.env.get_sensor_data('camera')
        possible_moves = [move for move in ACTIONS if self.env.check_if_move_possible(move)]
        possible_moves = self.avoid_repetition(possible_moves)
        
        if camera_data[0] == 3:  # Dog in front
            self.update_recent_moves('move_forward')
            return 'move_forward'
        elif 3 in camera_data[1:3]:  # Dog on the sides
            turn = 'turn_right' if camera_data[1] == 3 else 'turn_left'
            self.update_recent_moves(turn)
            return turn
        
        move = random.choice(possible_moves)
        self.update_recent_moves(move)
        return move
    
class RandomPolicy(ExpertPolicy):
    """
    We want to explore the reward function with some random moves
    """
    def act(self, state, training=False):
        possible_moves = [move for move in ACTIONS if self.env.check_if_move_possible(move)]
        possible_moves = self.avoid_repetition(possible_moves)
        move = random.choice(possible_moves)
        self.update_recent_moves(move)
        return move

class RecurrentNeuralNetworkRewardNetwork(nn.Module):
    """
        Initialize the RecurrentNeuralNetworkRewardNetwork.
        Parameters:
        - input_size: Size of the input features
        - hidden_size: Size of the hidden layer
        - num_actions: Number of possible actions
        - num_layers: Number of LSTM layers
        - seq_length: Length of the input sequence
        """
    def __init__(self, input_size=15, hidden_size=64, num_actions=3, num_layers=1, seq_length=15):
        super(RecurrentNeuralNetworkRewardNetwork, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length

        # Long Short-Term Memory (LSTM) layer to capture time dependencies in the memory sequence data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected  linear layer to map the hidden state to action values
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        """
        Forward pass through the LSTM and fully connected layer.
        The LSTM captures temporal dependencies in the input sequence.
        The final fully connected layer maps the hidden state to action values.
        Parameters:
        - x: Input tensor of shape (batch_size, seq_length, input_size)
        """
        # x shape: (batch_size, seq_length, input_size)
        _, (h_n, _) = self.lstm(x)  # LSTM output, hidden state, cell state
        # Use the last hidden state to predict action values
        out = self.fc(h_n[-1])
        return out

class ShortTermContextNetwork(nn.Module):
    def __init__(self, input_size=7):  # 4 for camera data, 3 for action encoding
        super(ShortTermContextNetwork, self).__init__()
        """
        Initialize the ShortTermContextNetwork.
        Parameters:
        - input_size: Size of the input features (4 for camera data, 3 for action encoding)
        """
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, 32)
        # Second fully connected layer
        self.fc2 = nn.Linear(32, 16)
        # Output layer with one neuron, safe or not safe
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        """
        Forward pass through the short-term context network.
        The network processes immediate sensor data and action encoding to predict safety.
        Parameters:
        - x: Input tensor of shape (batch_size, input_size)
        """
        # Apply ReLU activation after the first layer
        x = F.relu(self.fc1(x))
        # Apply ReLU activation after the second layer
        x = F.relu(self.fc2(x))
        # Output a probability score using sigmoid activation
        return torch.sigmoid(self.fc3(x))

class DualNetworkPolicy(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_actions=3):
        super(DualNetworkPolicy, self).__init__()
        """
        Initialize the DualNetworkPolicy.
        Parameters:
        - input_size: Size of the input features
        - hidden_size: Size of the hidden layer
        - num_actions: Number of possible actions
        """
        self.main_network = RecurrentNeuralNetworkRewardNetwork(input_size, hidden_size, num_actions)
        self.short_term_context_network = ShortTermContextNetwork()
        self.safety_threshold = 0.5 # ShortTermContextNetwork threshold of safety
    
    def to(self, device):
        """
        move model to certain device
        """
        self.main_network = self.main_network.to(device)
        self.short_term_context_network = self.short_term_context_network.to(device)
        return super().to(device)

    def forward(self, x):
        """
        forward pass on main_network
        """
        # x shape: (batch_size, seq_length, input_size)
        main_output = self.main_network(x)
        return main_output

    def select_safe_action(self, state, env: SimulationGridEnvironment):
        """
        checks action is safe
        """
        device = next(self.parameters()).device
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, SEQUENCE_LENGTH, -1).to(device)
        with torch.no_grad():
            action_values = self.forward(state_tensor)
            sorted_actions = torch.argsort(action_values, descending=True)[0]

        camera_data = env.get_sensor_data('camera')
        
        for action_index in sorted_actions:
            action = ACTIONS[action_index.item()]
            if env.check_if_move_possible(action):
                action_encoding = [0, 0, 0]
                action_encoding[ACTIONS.index(action)] = 1
                input_data = torch.tensor(np.concatenate([camera_data, action_encoding]), dtype=torch.float32).unsqueeze(0).to(device)
                safety_score = self.short_term_context_network(input_data).item()
                
                if safety_score >= self.safety_threshold:
                    return action

        # If no safe action is found, return the action with the highest safety score
        safety_scores = []
        for action in ACTIONS:
            action_encoding = [0, 0, 0]
            action_encoding[ACTIONS.index(action)] = 1
            input_data = torch.tensor(np.concatenate([camera_data, action_encoding]), dtype=torch.float32).unsqueeze(0).to(device)
            safety_scores.append(self.short_term_context_network(input_data).item())
        
        safest_action = ACTIONS[np.argmax(safety_scores)]
        return safest_action
     
class TrainingModel:
    def __init__(self, environment: SimulationGridEnvironment, hidden_size=64):
        """
        Initialize the TrainingModel.
        Parameters:
        - environment: The environment in which the model operates
        - hidden_size: Size of the hidden layer in the neural networks
        """
        self.env = environment
        self.vis_env = SimulationGridVisualisationEnvironment(size=environment.size)
        input_size = 15  # 4 directions, 4 camera inputs, 4 lidar inputs, 3 action possibilities
        self.main_loss_criterion = nn.MSELoss() # measure loss in mean squared error loss for main network
        self.short_term_context_loss_criterion = nn.BCELoss() # use binary cross entropy loss for short term context network
        self.losses = []
        self.losses_phase_two = []
        self.curriculum_losses = []

        # if gpu availiable use it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        # dual model is policy
        self.policy = DualNetworkPolicy(input_size, hidden_size, len(ACTIONS)).to(self.device)
        
        # use adam optimiser for training
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
    
    def train_on_batch(self, batch, curriculum=False):
        """
        Train the model on a batch of data.
        Parameters:
        - batch: Batch of data
        - curriculum: Boolean indicating if the training is part of curriculum learning
        """
        if curriculum or isinstance(batch[0], tuple):  # Check if it's curriculum data or second phase data
            # Convert the batch into tensors and reshape as necessary
            states = torch.stack([torch.tensor(item[0], dtype=torch.float32).view(SEQUENCE_LENGTH, -1) for item in batch]).to(self.device)
            actions = torch.tensor([item[1] for item in batch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(self.device)
        else:
            # Convert the batch into tensors 
            states, actions, rewards, _ = batch
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
        
        # Zero the gradients (reset to zero)
        self.optimizer.zero_grad()
        
        # Main network training
        main_outputs = self.policy.main_network(states)

        # gather the predicted rewards for the actions taken
        predicted_rewards = main_outputs.gather(1, actions.unsqueeze(1)).squeeze()
        main_loss = self.main_loss_criterion(predicted_rewards, rewards)
        
        # Short-context network training
        short_context_inputs = []
        for state, action in zip(states, actions):
            camera_data = state[0, :4]  # Take the first 4 elements (camera data) of the first time step
            action_encoding = torch.zeros(3, device=self.device)
            action_encoding[action] = 1
            short_context_input = torch.cat([camera_data, action_encoding])
            short_context_inputs.append(short_context_input)
        
        short_context_inputs = torch.stack(short_context_inputs).to(self.device)
        short_context_targets = (rewards > 0).float()
        short_context_output = self.policy.short_term_context_network(short_context_inputs)
        short_context_loss = self.short_term_context_loss_criterion(short_context_output.squeeze(), short_context_targets)
        
        # Combined loss
        total_loss = main_loss + short_context_loss
        total_loss.backward()
        self.optimizer.step()
        
        if curriculum:
            self.curriculum_losses.append(total_loss.item())
        else:
            self.losses_phase_two.append(total_loss.item())
        
        return total_loss.item()
    
    def get_policy(self):
        return self.policy

    def train(self, redo_short_context = False, num_iterations=1000, batch_size=16, second_phase=False, second_phase_iterations=0):
        """
        Train the model.
        Parameters:
        - num_iterations: Number of iterations for training
        - batch_size: Batch size for training
        - second_phase: Boolean indicating if second phase training should be performed
        - second_phase_iterations: Number of iterations for second phase training
        """
        self.losses = []
        db_handler = SQLiteDatabaseHandler(DB_PATH)
        batch_generator = db_handler.load_data_for_training(batch_size)
        
        for iteration in range(num_iterations):
            try:
                batch = next(batch_generator)
                # parse and convert rows to tensors
                states, actions, rewards, next_states = zip(*[self.parse_row(row) for row in batch])
                states = torch.tensor(np.array(states), dtype=torch.float32).reshape(batch_size, SEQUENCE_LENGTH, -1)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                loss = self.train_on_batch((states, actions, rewards, next_states))
                self.losses.append(loss)
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}, Loss: {loss}")
            except StopIteration:
                print("No more data to process.")
                break

        if redo_short_context:
                train_short_context_network(self, SimulationGridEnvironment(), batch_size=64)

        if second_phase:
            self.train_phase_two(second_phase_iterations, cats_count=3, dogs_count=3)

    def parse_row(self, row):
        """
        Parse a row of data from the database.
        Parameters:
        - row: Row of data from the database
        """
        _, state_sequence, action_index, reward, next_state = row
        state_sequence = np.array(eval(state_sequence))
        next_state = np.array(eval(next_state))
        return state_sequence, int(action_index), float(reward), next_state

    def generate_episode(self, env: SimulationGridEnvironment):
        """
        Generate an episode of interactions in the environment.
        Parameters:
        """
        episode_data = []
        done = False
        move_count = 0
        cat_count = len(env.cat_positions)
        while not done and move_count < 500 and env.dogs_reached == 0:

            if env.cats_reached == 3:
                good_job = True # run one more move after reached 3

            state = env.get_memory_state_sequence()
            policy = self.get_policy()

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, SEQUENCE_LENGTH, -1)
                action_values = policy(state_tensor)
                sorted_actions = torch.argsort(action_values, descending=True)[0].numpy()

            for action_index in sorted_actions:
                action = ACTIONS[action_index]
                if env.check_if_move_possible(action):
                    break

            next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=cat_count)
            action_index = ACTIONS.index(action)
            episode_data.append((state, action_index, reward, next_state))
            move_count += 1

            if good_job == True:
                done = True

        return episode_data, env.cats_reached


    @staticmethod
    def mp_generate_episode(env_params, model_params, state_dict, custom_positions=None, cats_count=3, dogs_count=3):
        """
        Generate an episode using multiprocessing.
        """
        try:
            env = SimulationGridEnvironment(**env_params)
            env.reset(custom_positions=custom_positions, cats_count=cats_count, dogs_count=dogs_count)
            
            # Create a DualNetworkPolicy instance and keep it on CPU
            policy = DualNetworkPolicy(**model_params)
            policy.load_state_dict(state_dict)
            policy.eval()
            
            # Explicitly move the policy to CPU
            policy = policy.cpu()

            episode_data = []
            done = False
            move_count = 0
            cat_count = len(env.cat_positions)
            times_completed = 0
            dogs_encountered = 0

            while move_count < 500 and env.dogs_reached == 0:
                if env.cats_reached == cat_count:
                    env.reset(custom_positions=custom_positions, cats_count=cats_count, dogs_count=dogs_count)
                    times_completed += 1
                    dogs_encountered = 0  # Reset dogs_encountered for the new episode

                try:
                    state = env.get_memory_state_sequence()
                    action = policy.select_safe_action(state, env)

                    if action is None:
                        print("No valid action found")
                        break

                    next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=cat_count)
                    action_index = ACTIONS.index(action)
                    episode_data.append((state, action_index, reward, next_state))
                    move_count += 1

                    if env.dogs_reached > dogs_encountered:
                        dogs_encountered += 1

                except Exception as e:
                    print(f"Error in episode loop: {e}")
                    break
                
            return episode_data, env.cats_reached + times_completed * cat_count, dogs_encountered
        except Exception as e:
            print(f"Error in mp_generate_episode: {e}")
            return [], 0, 0
            
    def parallel_generate_episodes(self, num_episodes, env_params, cats_count=3, dogs_count=3, timeout=60):
        """
        Generate multiple episodes in parallel using multiprocessing.
        Parameters:
        """
        model_params = {
            'input_size': self.policy.main_network.input_size,
            'hidden_size': self.policy.main_network.lstm.hidden_size,
            'num_actions': len(ACTIONS)
        }
        # Move model to CPU before getting state dict
        self.policy.cpu()
        state_dict = self.policy.state_dict()
        # Move model back to original device
        self.policy.to(self.device)

        with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:
            results = []
            custom_positions_list = [self.generate_custom_positions(env_params['size'], cats_count, dogs_count) for _ in range(num_episodes)]

            for custom_positions in custom_positions_list:
                result = pool.apply_async(self.mp_generate_episode, (env_params, model_params, state_dict, custom_positions, cats_count, dogs_count))
                try:
                    episode_data, cats_reached, dogs_encountered = result.get(timeout=timeout)
                    results.append((episode_data, cats_reached, dogs_encountered))
                except mp.TimeoutError:
                    print(f"Episode generation timed out after {timeout} seconds")
                except Exception as e:
                    print(f"Error generating episode: {e}")
            return results
        
    def generate_custom_positions(self, size, cats_count, dogs_count):
        """
        Generate custom positions for initializing the grid.
        """
        env = SimulationGridEnvironment(size=size)
        env.reset(cats_count=cats_count, dogs_count=dogs_count)
        return {
            'cat_positions': env.cat_positions,
            'dog_positions': env.dog_positions,
            'obstacle_positions': env.obstacle_positions,
            'robot_position': env.position,
            'robot_direction': env.direction
        }

    def train_phase_two(self, num_iterations=1000, batch_size=16, max_attempts=100, cats_count=3, dogs_count=3):
        """
        Undertake traditional RL for a certain amount of iterations starting with the model in its current state
        use max attempts to run in parrallel training on best runs
        """

        print("Starting phase two training...")
        self.losses_phase_two = []
        successful_episodes = 0
        new_data = []

        env_params = {'size': self.env.size}

        display = None
        clock = None

        while successful_episodes < num_iterations:
            should_visualise = (successful_episodes % 5 == 0)

            if should_visualise:
                print(f"Generating and visualising episode {successful_episodes}")
                if display is None:
                    pygame.init()
                    display = pygame.display.set_mode((self.vis_env.grid_width, self.vis_env.grid_height))
                    pygame.display.set_caption('Live Training')
                    clock = pygame.time.Clock()
                self.vis_env.reset(cats_count=cats_count, dogs_count=dogs_count)
                episode_data, cats_reached, dogs_encountered = self.generate_episode_with_visualisation(self.vis_env, display, clock)
                results = [(episode_data, cats_reached, dogs_encountered)]
            else:
                print(f"Generating {max_attempts} episodes using multiprocessing")
                results = self.parallel_generate_episodes(max_attempts, env_params, cats_count, dogs_count)

            if results:
                best_episode_data, best_cats_reached, best_dogs_encountered = max(results, key=lambda x: x[1])
                avg_cats_reached = sum([result[1] for result in results]) / len(results)
                avg_dogs_encountered = sum([result[2] for result in results]) / len(results)

                score_of_one_count = sum(1 for result in results if result[1] >= 1)
                negative_score_count = sum(1 for result in results if result[2] > 0)
                probability_of_cat_caught = score_of_one_count / len(results)
                probability_of_dog_caught = negative_score_count / len(results)

                if best_episode_data:
                    new_data.extend(best_episode_data)
                    successful_episodes += 1

                    if len(new_data) >= batch_size:
                        loss = self.train_on_batch(new_data[:batch_size], curriculum=False)
                        new_data = new_data[batch_size:]
                        self.losses_phase_two.append(loss)
                        
                        print(f"Phase Two - Episode {successful_episodes}, Best Cats Reached: {best_cats_reached}, "
                            f"Probability of cat Reached: {probability_of_cat_caught:.2f}, Probability of dog impact: {probability_of_dog_caught:.2f}, "
                            f"Average Cats Reached: {avg_cats_reached:.2f}, Average Dogs Encountered: {avg_dogs_encountered:.2f}, Loss: {loss:.4f}")
                    else:
                        print(f"Phase Two - Episode {successful_episodes}, Best Cats Reached: {best_cats_reached}, "
                            f"Probability of cat Reached: {probability_of_cat_caught:.2f}, Probability of dog impact: {probability_of_dog_caught:.2f}, "
                            f"Average Cats Reached: {avg_cats_reached:.2f}, Average Dogs Encountered: {avg_dogs_encountered:.2f}, Waiting for more data...")
            else:
                print("No valid episodes generated in this iteration")

        if display is not None:
            pygame.quit()
        print("Phase two training complete.")  

    def generate_episode_with_visualisation(self, env: SimulationGridVisualisationEnvironment, display, clock):
        """
        visualise in pygame
        """
        episode_data = []
        done = False
        move_count = 0
        cat_count = len(env.cat_positions)
        dogs_encountered = 0

        while not done and move_count < 500 and env.dogs_reached == 0 and env.cats_reached < cat_count:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return episode_data, env.cats_reached, dogs_encountered

            state = env.get_memory_state_sequence()
            action = self.policy.select_safe_action(state, env)

            next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=cat_count)
            if action in ACTIONS:
                action_index = ACTIONS.index(action)
                episode_data.append((state, action_index, reward, next_state))

            if env.dogs_reached > dogs_encountered:
                dogs_encountered += 1

            frame = env.get_frame()
            display.blit(frame, (0, 0))
            pygame.display.flip()
            clock.tick(10)  # 10 FPS

            move_count += 1

        pygame.time.wait(2000)  # Wait for 2 seconds at the end of the episode
        return episode_data, env.cats_reached, dogs_encountered
    
def evaluate_policy(env, policy, num_episodes=10, cats_needed=1, grid_size=6, cats_count=2, dogs_count=2):
    """
    evaluate quality of policy
    """
    print(f"Evaluating policy on a {grid_size}x{grid_size} grid with {cats_count} cats and {dogs_count} dogs...")
    video_filename = f'simulation_{grid_size}x{grid_size}_{cats_count}cats_{dogs_count}dogs_{cats_needed}needed.mp4'
    plot_filename = f'performance_plot_{grid_size}x{grid_size}_{cats_count}cats_{dogs_count}dogs_{cats_needed}needed.png'
    
    vis_env = SimulationGridVisualisationEnvironment(size=grid_size)
    frames = []
    performance = []
    dogs_encountered_list = []

    for episode in range(num_episodes):
        print(f"Evaluating Episode {episode + 1}/{num_episodes}")
        state = vis_env.reset(cats_count=cats_count, dogs_count=dogs_count)
        done = False
        move_count = 0
        dogs_encountered = 0

        while not done and move_count < 500:
            action = policy.select_safe_action(state, vis_env)
            state, reward, done = vis_env.step_reward_function(action, cats_needed=cats_needed)
            frames.append(vis_env.get_frame())
            move_count += 1
            
            # Check if a dog was encountered in this step
            if vis_env.dogs_reached > dogs_encountered:
                dogs_encountered += 1

        performance.append(vis_env.cats_reached - dogs_encountered * dogs_count)
        dogs_encountered_list.append(dogs_encountered)
        print(f"Episode {episode + 1} completed, Performance: {performance[-1]}, Dogs encountered: {dogs_encountered}")

    utils.save_video(frames, video_filename)
    utils.save_plot(performance, plot_filename)
    
    times_score_over_one = sum(1 for perf in performance if perf >= 1)
    probability_of_cat_caught = times_score_over_one / len(performance)
    probability_of_dog_encountered = sum(1 for dogs in dogs_encountered_list if dogs > 0) / len(dogs_encountered_list)
    average_performance = sum(performance) / len(performance)
    average_dogs_encountered = sum(dogs_encountered_list) / len(dogs_encountered_list)
    
    print(f"Probability of cat Reached: {probability_of_cat_caught:.2f}")
    print(f"Probability of dog Encountered: {probability_of_dog_encountered:.2f}")
    print(f"Average performance: {average_performance:.2f}")
    print(f"Average dogs encountered: {average_dogs_encountered:.2f}")

    print("Evaluation complete. Video and performance plot saved.")

def generate_expert_data(env: SimulationGridEnvironment, num_sequences=1000, load_previous=False):
    """
    Generate expert data by simulating expert policies in the environment.
    """

    print("Generating expert data...")
    realistic_expert = RealisticExpertPolicy(env)
    perfect_expert = ExpertPolicy(env)
    negative_expert = NegativeExpertPolicy(env)
    random_policy = RandomPolicy(env)
    
    db_handler = SQLiteDatabaseHandler(DB_PATH)
    print("Initializing database")
    db_handler.initialize_db()
    
    if not load_previous:
        print("Clearing database")
        db_handler.clear_db()
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM expert_data')
            existing_sequences = cursor.fetchone()[0]
        print(f'Database setup/loaded. Existing sequences: {existing_sequences}')
    except Exception as e:
        print(f"Error checking existing sequences: {e}")
        existing_sequences = 0
    
    sequences_to_generate = num_sequences - existing_sequences
    if sequences_to_generate > 0:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            for episode in range(sequences_to_generate):
                if episode % sequences_to_generate/10 == 0:
                    print(f"Generating episode {episode}/{sequences_to_generate}")
                scenario = random.choices(
                    [
                        {"cats": 1, "dogs": 1, "weight": 0.1},
                        {"cats": 2, "dogs": 1, "weight": 0.1},
                        {"cats": 2, "dogs": 2, "weight": 0.1},
                        {"cats": 2, "dogs": 3, "weight": 0.1},
                        {"cats": 3, "dogs": 3, "weight": 0.1},
                        {"cats": 1, "dogs": 2, "weight": 0.2},
                        {"cats": 1, "dogs": 3, "weight": 0.3},
                    ], 
                    weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]
                )[0]
                
                while True:
                    state = env.reset(cats_count=scenario["cats"], dogs_count=scenario["dogs"])
                    done = False
                    move_count = 0
                    episode_data = []

                    while not done and move_count < 100:
                        policy_choice = random.random()
                        if policy_choice < 0.5:
                            action = realistic_expert.act(state, training=True)
                        elif policy_choice < 0.7:
                            action = perfect_expert.act(state, training=True)
                        elif policy_choice < 0.9:
                            action = negative_expert.act(state, training=True)
                        else:
                            action = random_policy.act(state, training=True)

                        next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=scenario["cats"])
                        action_index = ACTIONS.index(action)
                        state_sequence = env.get_memory_state_sequence().tolist()
                        next_state_sequence = next_state.tolist()
                        episode_data.append((str(state_sequence), action_index, reward, str(next_state_sequence)))
                        state = next_state
                        move_count += 1

                    if env.cats_reached >= scenario["cats"]:
                        break

                for data in episode_data:
                    cursor.execute('''
                        INSERT INTO expert_data (state_sequence, action_index, reward, next_state)
                        VALUES (?, ?, ?, ?)
                    ''', data)
            conn.commit()
    
    print("Expert data generation complete.")

def make_model_from_expert_data(env, num_sequences, load_previous=False):
    """
    create model from expert data
    """
    generate_expert_data(env, num_sequences, load_previous)
    model = TrainingModel(env)
    return model

def create_fortress(env: SimulationGridEnvironment, quadrant):
    """
    stage in curriculum learning where a fortress of obstacles surrounds the cat
    with only one possible enterance
    """
    size = env.size
    half = size // 2
    
    # Define quadrant coordinates
    quadrants = {
        0: (0, 0, half-1, half-1),         # Top-left
        1: (half, 0, size-1, half-1),      # Top-right
        2: (0, half, half-1, size-1),      # Bottom-left
        3: (half, half, size-1, size-1),   # Bottom-right
        4: (half//2, 0, half+half//2-1, size-1),       # Middle vertical
        5: (0, half//2, size-1, half+half//2-1),       # Middle horizontal
        6: (half//2, half//2, half+half//2-1, half+half//2-1)  # Center
    }
    
    x1, y1, x2, y2 = quadrants[quadrant]
    
    # Create fortress walls
    fortress = [(x, y) for x in range(x1, x2+1) for y in range(y1, y2+1) 
                if x == x1 or x == x2 or y == y1 or y == y2]
    
    # Create entrance
    entrance_wall = random.choice(['top', 'bottom', 'left', 'right'])
    if entrance_wall == 'top':
        entrance = (random.randint(x1+1, x2-1), y1)
    elif entrance_wall == 'bottom':
        entrance = (random.randint(x1+1, x2-1), y2)
    elif entrance_wall == 'left':
        entrance = (x1, random.randint(y1+1, y2-1))
    else:  # right
        entrance = (x2, random.randint(y1+1, y2-1))
    
    fortress.remove(entrance)
    
    # Place cat
    cat_position = (random.randint(x1+1, x2-1), random.randint(y1+1, y2-1))
    while cat_position in fortress:
        cat_position = (random.randint(x1+1, x2-1), random.randint(y1+1, y2-1))
    
    return fortress, cat_position

def generate_curriculum_episode(env: SimulationGridEnvironment, expert: ExpertPolicy, stage):
    """
    generate episodes for curriculum learning
    """
    if stage.get('fortress', False):
        quadrant = random.randint(0, 6)  # 0-3: corners, 4-5: middle edges, 6: center
        fortress, cat_position = create_fortress(env, quadrant)
        env.obstacle_positions = fortress
        env.cat_positions = [cat_position]
        state = env.reset(cats_count=1, dogs_count=0, preserve_positions=True)
    else:
        state = env.reset(cats_count=stage['cats'], dogs_count=stage['dogs'])
        if stage['obstacles']:
            env.obstacle_positions = env.position_generation(min(5, env.size * env.size - stage['cats'] - stage['dogs'] - 1))
        else:
            env.obstacle_positions = []

    episode_data = []
    done = False
    move_count = 0

    while not done and move_count < 100:  # Increased max moves for fortress scenario
        action = expert.act(state, training=True)
        next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=stage['cats'])

        # for zero cats stages the robot cannot be done it must simply not die
        if stage['cats'] == 0:
            done = False

        action_index = ACTIONS.index(action)
        episode_data.append((state, action_index, reward, next_state))
        state = next_state
        move_count += 1

    return episode_data

def create_maze_path(env: SimulationGridEnvironment, start):
    """
    create path with only one viable way to go with at least two turns"""
    size = env.size
    maze = [[1 for _ in range(size)] for _ in range(size)]
    path = [start]
    maze[start[1]][start[0]] = 0

    def get_neighbors(x, y):
        return [(nx, ny) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                if 0 <= nx < size and 0 <= ny < size and maze[ny][nx] == 1]

    # Generate a path with at least two turns
    turns = 0
    current = start
    while turns < 2 or len(path) < size:
        neighbors = get_neighbors(*current)
        if not neighbors:
            break
        next_pos = max(neighbors, key=lambda pos: abs(pos[0] - start[0]) + abs(pos[1] - start[1]))
        maze[next_pos[1]][next_pos[0]] = 0
        path.append(next_pos)
        if current[0] != next_pos[0] and current[1] != next_pos[1]:
            turns += 1
        current = next_pos

    return maze, path

def generate_maze_episode(env: SimulationGridEnvironment, expert: ExpertPolicy):
    """
    stage in curriculum learning where there is only one viable path
    """
    size = env.size
    start = (random.randint(0, size-1), random.randint(0, size-1))

    maze, path = create_maze_path(env, start)
    end = path[-1]  # The furthest point becomes the cat's position

    # Place cat, and surround path with dogs or obstacles
    env.cat_positions = [end]
    env.dog_positions = []
    env.obstacle_positions = []

    for y in range(size):
        for x in range(size):
            if maze[y][x] == 1 and (x, y) not in path:
                neighbors = [(nx, ny) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                             if 0 <= nx < size and 0 <= ny < size and maze[ny][nx] == 0]
                if neighbors:
                    # Check if this position is in the direct path of the robot
                    if (x, y) in [(px+dx, py+dy) for px, py in path for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]]:
                        env.obstacle_positions.append((x, y))
                    else:
                        env.dog_positions.append((x, y))
                else:
                    env.obstacle_positions.append((x, y))

    state = env.reset(cats_count=1, dogs_count=len(env.dog_positions), preserve_positions=True)
    env.position = start

    episode_data = []
    done = False
    move_count = 0
    look_count = 0

    while not done and move_count < 100:  # Increased max moves for maze scenario
        # Every 3 moves, perform a look action
        if look_count % 3 == 0:
            for look_action in ['turn_left', 'turn_right', 'turn_right', 'turn_left']:
                action = look_action
                next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=1)
                action_index = ACTIONS.index(action)
                episode_data.append((state, action_index, reward, next_state))
                state = next_state
                if done:
                    break
            look_count = 0
        else:
            action = expert.act(state, training=True)
            next_state, reward, done = env.step_reward_function(action, training=True, cats_needed=1)
            action_index = ACTIONS.index(action)
            episode_data.append((state, action_index, reward, next_state))
            state = next_state
        
        move_count += 1
        look_count += 1

    return episode_data

def curriculum_training(model: TrainingModel, num_iterations=1000, batch_size=32):
    """
    Train the model using curriculum learning.
    Parameters:
    - model: The model to train
    - num_iterations: Number of iterations for training
    - batch_size: Batch size for training
    """
    print("Starting curriculum training...")
    stages = [
        {'cats': 3, 'dogs': 0, 'obstacles': False},
        {'cats': 2, 'dogs': 0, 'obstacles': False},
        {'cats': 1, 'dogs': 0, 'obstacles': False},
        {'cats': 3, 'dogs': 0, 'obstacles': True},
        {'cats': 2, 'dogs': 0, 'obstacles': True},
        {'cats': 1, 'dogs': 0, 'obstacles': True},
        {'cats': 1, 'dogs': 0, 'obstacles': True, 'fortress': True},
        {'cats': 1, 'dogs': 2, 'obstacles': True},
        {'cats': 1, 'dogs': 3, 'obstacles': True},
        {'maze': True},
    ]

    env = SimulationGridEnvironment(size=10)
    expert = ExpertPolicy(env)

    # Calculate iterations for each stage
    sequential_iterations = int(num_iterations * 0.85)
    iterations_per_stage = sequential_iterations // len(stages)
    random_iterations = num_iterations - sequential_iterations

    total_iterations = 0
    for stage_index, stage in enumerate(stages):
        print(f"\nTraining on stage {stage_index + 1}/{len(stages)}: {stage}")
        expert.mode = 'cat' if stage.get('cats', 0) > 0 else 'dog'
        
        stage_iterations = 0
        while stage_iterations < iterations_per_stage:
            episode_data = []
            for _ in range(batch_size):
                if stage.get('maze', False):
                    episode_data.extend(generate_maze_episode(env, expert))
                else:
                    episode_data.extend(generate_curriculum_episode(env, expert, stage))
            
            loss = model.train_on_batch(episode_data, curriculum=True)
            
            stage_iterations += 1
            total_iterations += 1
            
            if stage_iterations % 10 == 0:
                print(f"Stage {stage_index + 1}, Iteration {stage_iterations}, Total Iterations {total_iterations}, Loss: {loss}")
        
        print(f"Completed stage {stage_index + 1} after {stage_iterations} iterations")

    print("Starting random stage selection for the remaining iterations...")
    for _ in range(random_iterations):
        stage = random.choice(stages)
        expert.mode = 'cat' if stage.get('cats', 0) > 0 else 'dog'
        
        episode_data = []
        for _ in range(batch_size):
            if stage.get('maze', False):
                episode_data.extend(generate_maze_episode(env, expert))
            else:
                episode_data.extend(generate_curriculum_episode(env, expert, stage))
        
        loss = model.train_on_batch(episode_data, curriculum=True)
        total_iterations += 1
        
        if total_iterations % 10 == 0:
            print(f"Random Stage, Total Iterations {total_iterations}, Loss: {loss}")

    print(f"Curriculum training complete. Total iterations: {total_iterations}")
    return model.curriculum_losses

def generate_short_context_data(env: SimulationGridEnvironment, num_samples=1000):
    """
    Generate data for training the ShortTermContextNetwork.
    Curriculum training focus on cat obtaining, this can focus on dog avoiding
    """
     
    data = []
    for _ in range(num_samples):
        env.reset(dogs_count=env.size/3, cats_count=1, train_avoidance_network_mode = True)
        camera_data = env.get_sensor_data('camera')
        
        for action in ACTIONS:
            action_encoding = [0, 0, 0]
            action_encoding[ACTIONS.index(action)] = 1
            
            # Combine camera data and action encoding
            input_data = np.concatenate([camera_data, action_encoding])
            
            # Determine if the action is safe
            if action == 'move_forward' and camera_data[0] == 3:  # Dog in front
                safe = 0
            elif action in ['turn_left', 'turn_right'] and camera_data[0] == 3:  # Dog in front, but turning
                safe = 1
            elif action == 'move_forward' and camera_data[0] == 2:  # Cat in front
                safe = 1
            else:
                safe = 1
            
            data.append((input_data, safe))
    
    return data

# Update the train_short_context_network function
def train_short_context_network(model: TrainingModel, env: SimulationGridEnvironment, num_iterations=15000, batch_size=64):
    """
    Train the ShortTermContextNetwork.
    Parameters:
    - model: The model to train
    - env: The environment
    - num_iterations: Number of iterations for training
    - batch_size: Batch size for training
    """
    print("Starting dog avoidance network training...")
    optimizer = optim.Adam(model.policy.short_term_context_network.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for iteration in range(num_iterations):
        batch_data = generate_short_context_data(env, batch_size)
        inputs, targets = zip(*batch_data)
        
        inputs = torch.tensor(np.array(inputs), dtype=torch.float32).to(model.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(model.device)

        optimizer.zero_grad()
        outputs = model.policy.short_term_context_network(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 500 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item()}")

    print("Dog avoidance network training complete.")
    
    # Test the network
    test_data = generate_short_context_data(env, 1000)
    correct = 0
    for input_data, target in test_data:
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(model.device)
        output = model.policy.short_term_context_network(input_tensor)
        prediction = (output.item() >= 0.5)
        if prediction == target:
            correct += 1
    print(f"Dog avoidance network accuracy: {correct / len(test_data)}")


def main():
    """
    Main function to run the training and evaluation.
    """
    print('db glitch needs fixing, delete db prior to running each time for now')
    env = SimulationGridEnvironment(size=10)
    model = make_model_from_expert_data(env, num_sequences=10, load_previous=False)

    print("Model created, about to start curriculum training")
    curriculum_losses = curriculum_training(model, num_iterations=10000)
    
    print("Starting specialized dog avoidance training")
    train_short_context_network(model, env, batch_size=64)

    # Plot curriculum training losses
    plt.figure(figsize=(10, 5))
    plt.plot(curriculum_losses)
    plt.title('Curriculum Training Losses')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.savefig('curriculum_training_losses.png')
    plt.close()
    
    print("pre second stage evaluation")
    torch.save(model.get_policy().state_dict(), 'latest_curriculum_only_policy.pth')
    evaluate_policy(env, model.get_policy(), num_episodes=10)

    print("Curriculum training complete, starting regular training")
    model.train(num_iterations=10, batch_size=16, redo_short_context = False, second_phase=True, second_phase_iterations=10)

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

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
