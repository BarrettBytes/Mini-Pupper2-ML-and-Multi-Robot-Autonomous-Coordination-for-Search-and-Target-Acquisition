# Import necessary libraries
from datetime import datetime  # For handling date and time
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import matplotlib.image as mpimg  # For loading images
import random  # For random number generation
import tensorflow as tf  # For building and training neural networks
import cv2  # For image processing
import gc  # For garbage collection
from collections import deque  # For handling ordered collections

# Function to load and resize an image
def load_and_resize_image(image_path, target_size=(64, 64)):
    image = mpimg.imread(image_path)  # Load the image
    if image.shape[2] == 4:  # Check if the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
    if image.shape[:2] != target_size:  # Check if the image size is not the target size
        image = cv2.resize(image, target_size)  # Resize the image
    return image  # Return the resized image

# Function to generate random positions avoiding specified positions
def generate_positions(size, num_positions, min_distance=3, avoid=[]):
    positions = []
    while len(positions) < num_positions:
        position = (random.randint(0, size - 1), random.randint(0, size - 1))  # Generate a random position
        # Ensure the position is at least min_distance away from all positions in avoid
        if all(np.linalg.norm(np.array(position) - np.array(p)) >= min_distance for p in avoid):
            positions.append(position)
    return positions  # Return the generated positions

# Class representing the grid environment
class GridEnvironment:
    def __init__(self, size=10, dog_image_path='dog.png', cat_image_path='cat.png', robot_image_path='robot.png'):
        self.size = size  # Size of the grid
        self.dog_image = load_and_resize_image(dog_image_path)  # Load and resize dog image
        self.cat_image = load_and_resize_image(cat_image_path)  # Load and resize cat image
        self.robot_image = load_and_resize_image(robot_image_path)  # Load and resize robot image
        self.cell_height, self.cell_width, _ = self.dog_image.shape  # Dimensions of a single cell
        self.grid_height = self.cell_height * self.size  # Height of the entire grid
        self.grid_width = self.cell_width * self.size  # Width of the entire grid
        self.action_history = deque(maxlen=15)  # History of actions taken
        self.reset()  # Initialize the grid

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)  # Create an empty grid
        self.position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))  # Random starting position
        self.cat_positions = generate_positions(self.size, 3)  # Generate positions for cats
        self.dog_positions = generate_positions(self.size, 1, self.cat_positions)  # Generate a position for the dog
        self.found_cats = [False] * len(self.cat_positions)  # Track found cats
        self.moves = 0  # Initialize move counter
        self.cats_reached = 0  # Initialize counter for cats reached
        self.dogs_reached = 0  # Initialize counter for dogs reached
        self.last_seen = None  # Track last seen animal
        self.update_grid()  # Update the grid with initial positions
        return self.get_state()  # Return the initial state

    def update_grid(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)  # Create an empty grid
        self.grid[self.position] = 1  # Mark the robot's position
        for i, cat in enumerate(self.cat_positions):
            if not self.found_cats[i]:  # If the cat has not been found
                self.grid[cat] = 2  # Mark the cat's position
        for dog in self.dog_positions:
            self.grid[dog] = 3  # Mark the dog's position

    def step(self, action):
        self.moves += 1  # Increment move counter
        self.action_history.append(action)  # Record the action
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Possible moves: left, right, up, down
        next_position = (self.position[0] + moves[action][0], self.position[1] + moves[action][1])  # Calculate next position
        if 0 <= next_position[0] < self.size and 0 <= next_position[1] < self.size:
            self.position = next_position  # Update position if within bounds
        self.update_grid()  # Update the grid with new position
        reward, done = self.calculate_reward()  # Calculate reward and check if done
        if self.moves >= self.size * self.size * 10:
            done = True  # End the episode if max moves exceeded
        return self.get_state(), reward, done  # Return the new state, reward, and done flag

    def calculate_reward(self):
        if self.position in self.dog_positions:  # If the robot reaches a dog
            self.dogs_reached += 1
            return -1000, True  # Large negative reward and end episode
        for i, cat in enumerate(self.cat_positions):
            if self.position == cat and not self.found_cats[i]:  # If the robot reaches a cat
                self.found_cats[i] = True
                self.cats_reached += 1
                if all(self.found_cats):
                    return 5000, True  # Large reward if all cats found and end episode
                return 500 * self.cats_reached, False  # Reward for finding a cat
        return self.intermediate_reward() + self.loop_penalty(), False  # Calculate intermediate reward and loop penalty

    def intermediate_reward(self):
        camera_view = self.get_camera_data()  # Get the camera view data
        reward = -1  # Initialize reward
        if 2 in camera_view:
            cat_index = np.where(camera_view == 2)[0][0]
            reward += 50 / (cat_index + 1)  # Reward based on proximity to cat
            self.last_seen = 'cat'
        elif 3 in camera_view:
            dog_index = np.where(camera_view == 3)[0][0]
            reward -= 50 / (dog_index + 1)  # Penalty based on proximity to dog
            self.last_seen = 'dog'
        else:
            reward -= 10  # Penalize for no new information
            if self.last_seen == 'cat':
                reward -= 100  # Penalty if last seen was a cat
            elif self.last_seen == 'dog':
                reward += 100  # Reward if last seen was a dog
            self.last_seen = None
        return reward  # Return the calculated reward

    def loop_penalty(self):
        if len(self.action_history) < 15:
            return 0  # No penalty if action history is short
        recent_actions = list(self.action_history)[-12:]  # Check the last 12 actions
        tortoise = hare = 0
        while hare < len(recent_actions) - 1:
            if recent_actions[tortoise] == recent_actions[hare]:
                return -50  # Penalty for repeating actions
            tortoise += 1
            hare += 2
        return 0  # No penalty if no loops detected

    def get_state(self):
        state = self.grid.flatten() / 3  # Normalize grid values to [0, 1]
        camera_data = self.get_camera_data() / 3  # Normalize camera data
        return np.concatenate((state, camera_data))  # Concatenate grid state and camera data

    def get_camera_data(self):
        camera_view = np.zeros(self.size, dtype=np.uint8)  # Initialize camera view
        x, y = self.position
        for i in range(x, self.size):
            if self.grid[i, y] == 2:
                camera_view[i - x] = 2  # Mark cat in camera view
                break
            elif self.grid[i, y] == 3:
                camera_view[i - x] = 3  # Mark dog in camera view
                break
        return camera_view  # Return the camera view data

# Class representing the Proximal Policy Optimization (PPO) agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, clip_ratio=0.2, update_steps=10, buffer_size=2048, epsilon=1.0, epsilon_decay=0.995):
        self.state_dim = state_dim  # Dimension of the state space
        self.action_dim = action_dim  # Dimension of the action space
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.clip_ratio = clip_ratio  # Clipping ratio for PPO
        self.update_steps = update_steps  # Number of steps for updating
        self.buffer_size = buffer_size  # Size of the experience buffer
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.actor = self.build_actor()  # Build the actor network
        self.critic = self.build_critic()  # Build the critic network
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)  # Optimizer for the actor
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)  # Optimizer for the critic
        self.memory = deque(maxlen=self.buffer_size)  # Experience buffer

    def build_actor(self):
        # Build the actor network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),  # Hidden layer
            tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
            tf.keras.layers.Dense(self.action_dim, activation='softmax')  # Output layer
        ])
        return model  # Return the actor network

    def build_critic(self):
        # Build the critic network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),  # Hidden layer
            tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
            tf.keras.layers.Dense(1)  # Output layer
        ])
        return model  # Return the critic network

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:  # Exploration
            return np.random.randint(self.action_dim)  # Random action
        else:  # Exploitation
            state = np.reshape(state, [1, self.state_dim])
            probabilities = self.actor(state, training=False)
            return np.random.choice(self.action_dim, p=probabilities.numpy()[0])  # Action based on policy

    def store_transition(self, transition):
        self.memory.append(transition)  # Store the transition in memory

    def update(self):
        if len(self.memory) < self.buffer_size:
            return  # Wait until buffer is full

        # Extract states, actions, rewards, dones, and next_states from memory
        states, actions, rewards, dones, next_states = zip(*self.memory)
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.vstack(next_states)

        # Calculate advantages and target values
        advantages, target_values = self.calculate_advantages_and_targets(states, rewards, dones, next_states)
        self.update_actor(states, actions, advantages)  # Update the actor network
        self.update_critic(states, target_values)  # Update the critic network
        self.memory.clear()  # Clear the memory

        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)  # Decay epsilon

    def calculate_advantages_and_targets(self, states, rewards, dones, next_states):
        values = self.critic(states).numpy()
        next_values = self.critic(next_states).numpy()

        advantages = np.zeros_like(rewards, dtype=np.float32)  # Initialize advantages
        targets = np.zeros_like(rewards, dtype=np.float32)  # Initialize target values
        running_advantage = 0.0  # Initialize running advantage

        # Calculate advantages and targets using GAE (Generalized Advantage Estimation)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_advantage = 0.0  # Reset running advantage on episode end
                running_target = rewards[t]  # Set running target to reward
            else:
                running_target = rewards[t] + self.gamma * next_values[t] * (1 - dones[t])
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
                running_advantage = delta + self.gamma * self.clip_ratio * running_advantage * (1 - dones[t])

            advantages[t] = running_advantage  # Update advantages
            targets[t] = running_target  # Update targets

        return advantages, targets  # Return advantages and targets

    def update_actor(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.actor(states, training=True)
            action_probs = tf.nn.softmax(logits)
            action_indices = np.arange(len(actions)) * self.action_dim + actions
            action_probs = tf.gather(tf.reshape(action_probs, [-1]), action_indices)
            loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-10) * advantages)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))  # Apply gradients to actor

    def update_critic(self, states, targets):
        targets = np.reshape(targets, (-1, 1))
        with tf.GradientTape() as tape:
            values = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(targets - values))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))  # Apply gradients to critic

# Function to simulate the environment and agent
def simulate(agent, environment, episodes, video_filename='grid_simulation.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define video codec
    fps = 10  # Frames per second
    out = cv2.VideoWriter(video_filename, fourcc, fps, (640, 640))  # Video writer

    save_indices = np.linspace(0, episodes - 1, 5, dtype=int)  # Episodes to save

    total_cats_reached = 0  # Initialize total cats reached
    total_dogs_reached = 0  # Initialize total dogs reached
    total_moves = 0  # Initialize total moves
    episodes_since_last_print = 0  # Initialize episodes since last print

    for episode in range(episodes):
        state = environment.reset()  # Reset the environment
        done = False
        while not done:
            action = agent.select_action(state)  # Select an action
            next_state, reward, done = environment.step(action)  # Execute the action
            agent.store_transition((state, action, reward, done, next_state))  # Store the transition
            state = next_state  # Update state

            if episode in save_indices:  # Save frames for specific episodes
                fig, ax = plt.subplots(figsize=(6, 6))
                grid_image = np.ones((environment.grid_height, environment.grid_width, 3))

                for dog_pos in environment.dog_positions:
                    x0, y0 = dog_pos[1] * environment.cell_width, dog_pos[0] * environment.cell_height
                    x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
                    grid_image[y0:y1, x0:x1] = environment.dog_image

                for cat_pos in environment.cat_positions:
                    if not environment.found_cats[environment.cat_positions.index(cat_pos)]:
                        x0, y0 = cat_pos[1] * environment.cell_width, cat_pos[0] * environment.cell_height
                        x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
                        grid_image[y0:y1, x0:x1] = environment.cat_image

                x0, y0 = environment.position[1] * environment.cell_width, environment.position[0] * environment.cell_height
                x1, y1 = x0 + environment.cell_width, y0 + environment.cell_height
                grid_image[y0:y1, x0:x1] = environment.robot_image

                ax.imshow(grid_image, interpolation='nearest', aspect='auto')
                ax.set_title(f"Episode {episode + 1}")
                ax.axis('off')

                fig.canvas.draw()

                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                img_resized = cv2.resize(img, (640, 640))

                out.write(img_resized)  # Write frame to video
                plt.close(fig)

        total_cats_reached += environment.cats_reached  # Update total cats reached
        total_dogs_reached += environment.dogs_reached  # Update total dogs reached
        total_moves += environment.moves  # Update total moves
        episodes_since_last_print += 1

        if episode in save_indices:  # Print averages for specific episodes
            avg_cats_reached = total_cats_reached / episodes_since_last_print
            avg_dogs_reached = total_dogs_reached / episodes_since_last_print
            avg_moves = total_moves / episodes_since_last_print

            print(f"Episode {episode + 1} complete. Cats reached: {environment.cats_reached}, Dogs reached: {environment.dogs_reached}, Moves: {environment.moves}")
            print(f"Average since last print - Cats reached: {avg_cats_reached}, Dogs reached: {avg_dogs_reached}, Moves: {avg_moves}")

            total_cats_reached = 0
            total_dogs_reached = 0
            total_moves = 0
            episodes_since_last_print = 0

            for _ in range(5):
                out.write(img_resized)  # Write frames to video

        agent.update()  # Update the agent

        if episode % 10 == 0:
            gc.collect()  # Run garbage collection every 10 episodes

    out.release()  # Release the video writer
    print(f"Video saved as {video_filename}")

# Main function to set up and run the simulation
if __name__ == "__main__":
    grid_size = 5  # Size of the grid
    episodes = 6000  # Number of episodes to train

    dog_image_path = 'dog.png'  # Path to dog image
    cat_image_path = 'cat.png'  # Path to cat image
    robot_image_path = 'robot.png'  # Path to robot image

    env = GridEnvironment(size=grid_size, dog_image_path=dog_image_path, cat_image_path=cat_image_path, robot_image_path=robot_image_path)  # Initialize environment
    agent = PPOAgent(state_dim=grid_size**2 + grid_size, action_dim=4, lr=0.005, gamma=0.99, clip_ratio=0.2, update_steps=10, buffer_size=2048, epsilon_decay=0.01)  # Initialize agent
    date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Get current date and time

    simulate(agent, env, episodes, video_filename='grid_simulation_' + date_time + '.mp4')  # Run the simulation
