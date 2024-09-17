# Robot Training Framework README

This RL section contains code that trains a robot to find cats and avoid obstacles using a structured environment, expert policies, reinforcement learning, and inverse reinforcement learning.

## 1. Environment Setup: `GridEnvironment`

The `GridEnvironment` class sets up the grid where the robot operates, including cats, dogs (obstacles), and the robot itself.

### Key Functions:

- **`__init__()`**: Initializes the environment, loads images for the robot, cats, and dogs, and sets up the grid.
  ```python
  def __init__(self, size=6, dog_image_path='dog.png', cat_image_path='cat.png', robot_image_path='robot.png'):
      # Load images and initialize the grid
  ```

- **`reset()`**: Resets the environment, placing cats, dogs, and the robot in new positions. Can preserve positions and states if needed.
  ```python
  def reset(self, training=False, preserve_positions=False, cats_count=2, dogs_count=2, preserve_cats_found=False):
      # Reset the positions and states in the environment
  ```

- **`generate_positions()`**: Generates random positions for cats and dogs while ensuring a minimum distance between them.
  ```python
  def generate_positions(self, num_positions, min_distance=2, avoid=[]):
      # Generate random positions for objects
  ```

## 2. Sensing Mechanisms: `get_camera_data` and `get_lidar_data`

The robot uses camera and LIDAR data to navigate the environment.

### Key Functions:

- **`get_camera_data()`**: Collects data from the robot's camera, identifying the presence of cats and dogs in the four cardinal directions and the current position.
  ```python
  def get_camera_data(self):
      # Return the camera view
  ```

- **`get_lidar_data()`**: Uses LIDAR to detect obstacles (dogs) in the environment, providing a distance measure to the nearest obstacle in each direction.
  ```python
  def get_lidar_data(self):
      # Return the LIDAR view
  ```

## 3. Movement and Interaction: `step` Function

The `step` function updates the robot's position based on the chosen action, calculates the reward, and checks if the episode is done.

### Key Functions:

- **`step()`**: Executes an action, updates the environment, and computes the reward. The robot gets positive rewards for finding cats and negative rewards for encountering dogs or obstacles.
  ```python
  def step(self, action, training=False, cats_needed=1):
      # Move the robot, update the environment, and calculate rewards
  ```

## 4. Expert Policies

These classes define how the robot should act based on the current state, using different strategies.

### Key Classes:

- **`ExpertPolicy`**: A perfect policy where the robot moves directly towards visible cats and away from dogs.
  ```python
  class ExpertPolicy(AntiRepetitionMixin):
      def act(self, state, training=False):
          # Determine the best action based on camera data
  ```

- **`RealisticExpertPolicy`**: A more realistic policy that includes human-like errors and memory.
  ```python
  class RealisticExpertPolicy(AntiRepetitionMixin):
      def act(self, state, training=False):
          # Determine the best action with a chance of error
  ```

## 5. Data Management and Training: `generate_expert_data` and `IRLModel`

The training data is generated using expert policies, and then an inverse reinforcement learning model is trained to learn these policies.

### Key Functions and Classes:

- **`generate_expert_data()`**: Uses expert policies to generate and store sequences of states, actions, and rewards in a SQLite database.
  ```python
  def generate_expert_data(env: GridEnvironment, num_sequences=1000, load_previous=False):
      # Generate data using expert policies and store in the database
  ```

- **`IRLModel`**: Implements an inverse reinforcement learning model using an LSTM network to predict rewards and learn the optimal policy.
  ```python
  class IRLModel:
      def __init__(self, environment, hidden_size=64):
          # Initialize the IRL model with LSTM
      def train(self, num_iterations=1000, batch_size=16, second_phase=False, second_phase_iterations=0):
          # Train the IRL model on the expert data
  ```

## 6. Policy Evaluation: `evaluate_policy`

Finally, the learned policy is evaluated to check its performance in finding cats and avoiding obstacles.

### Key Function:

- **`evaluate_policy()`**: Runs the learned policy in the environment and evaluates its performance, generating a video and plot of the results.
  ```python
  def evaluate_policy(env: GridEnvironment, policy, num_episodes=10, cats_needed=1, grid_size=6, cats_count=2, dogs_count=2):
      # Evaluate the learned policy and save the results
  ```

## Summary

The code achieves the goal of training a robot to find cats and avoid obstacles through the following steps:

1. **Environment Setup**: Defines the grid and places objects randomly.
2. **Sensing Mechanisms**: Uses camera and LIDAR to gather data about the surroundings.
3. **Movement and Interaction**: Implements the `step` function to move the robot and calculate rewards.
4. **Expert Policies**: Uses predefined strategies to generate training data.
5. **Data Management and Training**: Generates and stores expert data, then trains an IRL model to learn the optimal policy.
6. **Policy Evaluation**: Evaluates the learned policy to ensure it effectively finds cats and avoids obstacles.

## Why This Works

1. **Structured Environment**: A well-defined grid environment simplifies the problem and ensures manageable interactions.
2. **Expert Policies**: High-quality training data from optimal and realistic behaviors guide the robot's learning.
3. **Reinforcement Learning**: Reward-based learning and sequence learning with RNNs capture complex temporal dependencies.
4. **Inverse Reinforcement Learning**: Learning the reward function helps in generalizing to new scenarios.
5. **Data Management**: Efficient data storage and management support scalable and robust training.
6. **Systematic Evaluation**: Continuous assessment and feedback ensure the model's effectiveness.

By integrating these components, the framework leverages both structured expert knowledge and powerful learning algorithms to train the robot effectively.
