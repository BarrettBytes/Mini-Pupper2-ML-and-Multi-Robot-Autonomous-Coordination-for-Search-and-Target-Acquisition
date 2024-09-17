import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
import numpy as np
import torch
import Demo2  # Importing Demo2 which contains the model definition
import os
import time
from collections import deque

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SEQ_LENGTH = 30  # Length of the action and sensor history

class ImprovedNNInferenceNode(Node):
    def __init__(self):
        super().__init__('improved_nn_inference_node')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/yolov4_detections',
            self.detection_callback,
            10
        )
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Define model parameters
        input_size = 5 + len(ACTIONS)
        hidden_size = 64
        num_actions = len(ACTIONS)
        
        # Initialize and load the trained model
        self.model = Demo2.SimpleRNNRewardNetwork(input_size, hidden_size, num_actions)
        self.model.load_state_dict(torch.load('learned_policy.pth'))
        self.model.eval()

        # Initialize action history
        self.action_history = deque(maxlen=SEQ_LENGTH)
        for _ in range(SEQ_LENGTH):
            self.action_history.append(np.zeros(len(ACTIONS)))

        # State management
        self.state = 'wait_for_detection'
        self.command = Twist()
        self.direction = 'unknown'
        self.action_start_time = None
        self.turn_duration = 4.5  # seconds for 90-degree rotation actions
        self.backward_turn_duration = 9.0  # seconds for 180-degree rotation (backward action)
        self.move_duration = 8.0  # seconds for forward movement
        self.wait_duration = 6.0  # seconds to wait between actions
        self.create_timer(0.1, self.timer_callback)  # 10 Hz timer for state machine
        
        self.start_time = time.time()
        self.last_detection_time = time.time()

        # Load COCO labels
        self.coco_labels = self.load_coco_labels('coco.names')

        # Timer for connection checking
        self.create_timer(30.0, self.check_detection_timer_callback)

        # Detection counters
        self.cat_detections = 0
        self.dog_detections = 0
        self.total_detections = 0

        # Last movement direction
        self.last_movement_direction = None

    def load_coco_labels(self, filepath):
        with open(filepath, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def detection_callback(self, msg):
        self.last_detection_time = time.time()
        self.process_detections(msg)

    def process_detections(self, detections_msg):
        for detection in detections_msg.detections:
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                if score > 0.5:
                    self.total_detections += 1
                    if class_id == 15 or class_id == 77:  # Cat or teddy bear (treated as cat)
                        self.cat_detections += 1
                        print('Cat detected!')
                    elif class_id == 16:  # Dog
                        self.dog_detections += 1
                        print('Dog detected!')

    def timer_callback(self):
        current_time = time.time()

        if self.state == 'wait_for_detection':
            if self.total_detections > 0:
                self.process_accumulated_detections()
        elif self.state == 'start_action':
            if self.direction == 'forward':
                self.move_forward()
                self.state = 'moving'
            elif self.direction == 'backward':
                self.turn()  # Start 180-degree rotation
                self.state = 'turning_backward'
            else:
                self.turn()
                self.state = 'turning'
            self.action_start_time = current_time
        elif self.state == 'turning':
            if current_time - self.action_start_time >= self.turn_duration:
                self.stop_action()
                self.state = 'move_after_turn'
                self.action_start_time = current_time
        elif self.state == 'turning_backward':
            if current_time - self.action_start_time >= self.backward_turn_duration:
                self.stop_action()
                self.state = 'move_after_backward_turn'
                self.action_start_time = current_time
        elif self.state == 'move_after_turn' or self.state == 'move_after_backward_turn':
            if current_time - self.action_start_time >= self.wait_duration:
                self.move_forward()
                self.state = 'moving'
                self.action_start_time = current_time
        elif self.state == 'moving':
            if current_time - self.action_start_time >= self.move_duration:
                self.stop_action()
                self.state = 'waiting'
                self.action_start_time = current_time
        elif self.state == 'waiting':
            if current_time - self.action_start_time >= self.wait_duration:
                self.state = 'wait_for_detection'

    def process_accumulated_detections(self):
        if self.total_detections == 0:
            return

        detection_features = np.zeros(5)
        if self.cat_detections > 0 or self.dog_detections > 0:
            if self.cat_detections > self.dog_detections:
                if self.last_movement_direction is not None:
                    detection_features[self.last_movement_direction] = 2
                print("Processing cat detection")
            elif self.dog_detections > self.cat_detections:
                if self.last_movement_direction is not None:
                    detection_features[self.last_movement_direction] = 3
                print("Processing dog detection")

        # Reset detection counters
        self.cat_detections = 0
        self.dog_detections = 0
        self.total_detections = 0

        # Create input tensor and process model output
        input_tensor = self.create_input_tensor(detection_features)
        self.process_model_output(input_tensor)
        self.state = 'start_action'
        self.action_start_time = time.time()

    def create_input_tensor(self, detection_features):
        detection_sequence = np.tile(detection_features, (SEQ_LENGTH, 1))
        action_history = self.get_action_history().reshape(SEQ_LENGTH, -1)
        input_features = np.concatenate((detection_sequence, action_history), axis=1)
        return torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    def turn(self):
        if self.direction == 'left':
            self.command.angular.z = 1.0
        elif self.direction == 'right':
            self.command.angular.z = -1.0
        elif self.direction == 'backward':
            self.command.angular.z = 1.0  # Turn left for backward (180-degree rotation)
        self.publisher_.publish(self.command)
        print(f'Turning: {self.direction}')

    def move_forward(self):
        forward_command = Twist()
        forward_command.linear.x = 0.5
        self.publisher_.publish(forward_command)
        print('Moving forward')

    def stop_action(self):
        stop_command = Twist()
        self.publisher_.publish(stop_command)
        print('Stopping action')

    def get_action_history(self):
        return np.array(list(self.action_history)).flatten()

    def process_model_output(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        self.process_output(output)
        action_index = torch.argmax(output, dim=1).item()
        new_action = np.zeros(len(ACTIONS))
        new_action[action_index] = 1.0
        self.action_history.append(new_action)
        self.last_movement_direction = action_index + 1  # Update last movement direction

    def process_output(self, output):
        sorted_actions = torch.argsort(output, dim=1, descending=True)[0]
        for action_index in sorted_actions:
            action = ACTIONS[action_index.item()]
            if self.is_move_possible(action):
                self.direction = ['forward', 'backward', 'left', 'right'][action_index.item()]
                self.command = self.action_to_twist(action)
                break
        print(f"Chosen action: {self.direction}")

    def action_to_twist(self, action):
        twist = Twist()
        if action == (0, -1):  # Forward
            twist.linear.x = 0.5
        elif action == (0, 1):  # Backward
            twist.angular.z = 1.0
        elif action == (-1, 0):  # Left
            twist.angular.z = 1.0
        elif action == (1, 0):  # Right
            twist.angular.z = -1.0
        return twist

    def is_move_possible(self, action):
        # This is a placeholder. Implement based on your robot's capabilities and environment
        return True

    def check_detection_timer_callback(self):
        current_time = time.time()
        if current_time - self.last_detection_time > 30:
            print('No detection data received for 30 seconds. Still trying to connect...')

def main(args=None):
    rclpy.init(args=args)
    improved_nn_inference_node = ImprovedNNInferenceNode()
    rclpy.spin(improved_nn_inference_node)
    improved_nn_inference_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
