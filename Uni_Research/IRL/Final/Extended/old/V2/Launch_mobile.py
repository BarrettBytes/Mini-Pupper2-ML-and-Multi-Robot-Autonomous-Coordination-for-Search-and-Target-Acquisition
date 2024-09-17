# for this you need to use  python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py instead of  python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart.py

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
import Demo2  # Importing Demo2 which contains the model definition
import os
import time
from collections import deque
from scipy.ndimage import median_filter
import threading
from datetime import datetime

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SEQ_LENGTH = 30  # Length of the action and sensor history

class PauseManager:
    def __init__(self):
        self.paused = False
        self.lock = threading.Lock()

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def is_paused(self):
        with self.lock:
            return self.paused

class ImprovedNNInferenceNode(Node):
    def __init__(self, pause_manager):
        super().__init__('improved_nn_inference_node')
        self.pause_manager = pause_manager
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/mobilenet_detections',
            self.detection_callback,
            10
        )
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        self.image_subscription = self.create_subscription(
            Image,
            '/color/image',
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Define model parameters
        input_size = 14  # Ensuring the total input size matches the original model
        hidden_size = 64
        num_actions = len(ACTIONS)
        
        # Initialize and load the trained model
        self.model = Demo2.SimpleRNNRewardNetworkWithAttention(input_size, hidden_size, num_actions)
        self.model.load_state_dict(torch.load('learned_policy.pth'))
        self.model.eval()

        # Initialize action history
        self.action_history = deque(maxlen=SEQ_LENGTH)
        for _ in range(SEQ_LENGTH):
            self.action_history.append(np.zeros(len(ACTIONS)))

        # State management
        self.state = 'wait_for_message'
        self.command = Twist()
        self.direction = 'unknown'
        self.action_start_time = None
        self.action_complete = True
        self.turn_duration = 4.5  # seconds for 90-degree rotation actions
        self.backward_turn_duration = 9.0  # seconds for 180-degree rotation (backward action)
        self.move_duration = 8.0  # seconds for forward movement
        self.wait_duration = 6.0  # seconds to wait between actions
        self.create_timer(0.1, self.timer_callback)  # 10 Hz timer for state machine

        self.start_time = time.time()
        self.last_detection_time = time.time()

        # Load PASCAL VOC labels
        self.mobilenet_labels = self.load_mobilenet_labels()

        # Timer for connection checking
        self.create_timer(30.0, self.check_detection_timer_callback)
        self.waiting_message_printed = False

        # Detection counters
        self.cat_detections = 0
        self.dog_detections = 0
        self.total_detections = 0

        # Last movement direction
        self.last_movement_direction = None

        # LiDAR data storage
        self.lidar_data = np.zeros(5)
        self.lidar_data_history = deque(maxlen=5)  # For temporal filtering

        self.was_paused = False  # Track previous pause state to avoid redundant stop actions

        # Video creation
        self.frame_width = 640
        self.frame_height = 480
        self.frame_rate = 1.0
        self.last_valid_frame = None
        self.last_frame_time = 0
        self.segment_index = 0
        self.combined_index = 0
        self.current_segment = None
        self.output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_recording = True  # Start recording immediately


        print("Node initialized. Press 'Space' to pause/resume.")

    def load_mobilenet_labels(self):
        return [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
        ]

    def detection_callback(self, msg):
        if self.pause_manager.is_paused():
            return
        self.last_detection_time = time.time()
        self.process_detections(msg)
        if self.state == 'wait_for_message':
            self.state = 'process_action'
            self.action_complete = True

    def lidar_callback(self, msg):
        if self.pause_manager.is_paused():
            return
        ranges = np.array(msg.ranges)
        ranges = median_filter(ranges, size=3)  # Apply median filter

        current_data = np.zeros(5)
        front_indices = np.where((ranges >= 0.10) & (ranges < 0.30))  # Adjusted for front detection
        left_indices = np.where(ranges < 0.15)  # Adjusted for left detection
        right_indices = np.where(ranges < 0.15)  # Adjusted for right detection
        back_indices = np.where(ranges < 0.20)  # Adjusted for back detection

        if np.any(ranges[front_indices]):
            current_data[0] = 1
        if np.any(ranges[left_indices]):
            current_data[2] = 1
        if np.any(ranges[right_indices]):
            current_data[3] = 1
        if np.any(ranges[back_indices]):
            current_data[1] = 1

        self.lidar_data_history.append(current_data)
        self.lidar_data = np.mean(self.lidar_data_history, axis=0) > 0.5

    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_frame_time < 1.0:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frame(cv_image)
        self.last_frame_time = current_time

    def process_frame(self, frame):
        if self.pause_manager.is_paused():
            return

        if frame is None or frame.size == 0 or np.max(frame) == 0:
            if self.last_valid_frame is not None:
                frame = self.last_valid_frame.copy()
            else:
                print("Warning: No valid frame to write")
                return
        else:
            self.last_valid_frame = frame.copy()

        if frame.shape[0] != self.frame_height or frame.shape[1] != self.frame_width:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        if self.current_segment is None:
            self.start_new_segment()

        self.current_segment.write(frame)

    def start_new_segment(self):
        self.segment_index += 1
        segment_filename = os.path.join(self.output_dir, f'segment_{self.segment_index}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.current_segment = cv2.VideoWriter(
            segment_filename,
            fourcc,
            self.frame_rate,
            (self.frame_width, self.frame_height)
        )
        print(f'Started new video segment: {segment_filename}')

    def toggle_pause(self):
        if self.pause_manager.is_paused():
            self.resume_recording()
        else:
            self.pause_recording()
        self.pause_manager.toggle_pause()

    def pause_recording(self):
        if self.current_segment is not None:
            self.current_segment.release()
            self.current_segment = None
            self.create_combined_video()
        print("Recording paused")

    def resume_recording(self):
        self.start_new_segment()
        if self.last_valid_frame is not None:
            self.current_segment.write(self.last_valid_frame)
        print("Recording resumed")

    def create_combined_video(self):
        if self.segment_index < 2:
            print("Not enough segments to create a combined video")
            return

        self.combined_index += 1
        combined_filename = os.path.join(self.output_dir, f'combined_{self.combined_index}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_video = cv2.VideoWriter(
            combined_filename,
            fourcc,
            self.frame_rate,
            (self.frame_width, self.frame_height)
        )

        if self.combined_index == 1:
            # Combine the first two segments
            for i in range(1, 3):
                segment = cv2.VideoCapture(os.path.join(self.output_dir, f'segment_{i}.mp4'))
                self.append_video(segment, combined_video)
        else:
            # Combine the previous combined video with the latest segment
            prev_combined = cv2.VideoCapture(os.path.join(self.output_dir, f'combined_{self.combined_index-1}.mp4'))
            self.append_video(prev_combined, combined_video)

            latest_segment = cv2.VideoCapture(os.path.join(self.output_dir, f'segment_{self.segment_index}.mp4'))
            self.append_video(latest_segment, combined_video)

        combined_video.release()
        print(f'Created new combined video: {combined_filename}')

    def append_video(self, source, destination):
        while True:
            ret, frame = source.read()
            if not ret:
                break
            destination.write(frame)
        source.release()

    def finalize_videos(self):
        if self.current_segment is not None:
            self.current_segment.release()
            self.create_combined_video()
        print("All videos finalized")

    def __del__(self):
        self.finalize_videos()
        
    def timer_callback(self):
        if self.pause_manager.is_paused():
            if not self.was_paused:
                self.stop_action()
                self.finalize_videos()
                self.was_paused = True
            return
        else:
            if self.was_paused:
                self.create_combined_video()
            self.was_paused = False

        current_time = time.time()

        if self.state == 'wait_for_message':
            if not self.waiting_message_printed:
                print("Waiting for messages...")
                self.waiting_message_printed = True

        elif self.state == 'process_action' and self.action_complete:
            self.process_accumulated_detections()

        elif self.state == 'start_action':
            self.action_complete = False
            if self.direction == 'forward':
                self.move_forward()
                self.state = 'moving'
            elif self.direction == 'backward':
                self.turn()
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
                self.state = 'wait_for_message'
                self.action_complete = True

    def process_detections(self, detections_msg):
        self.total_detections = len(detections_msg.detections)
        for detection in detections_msg.detections:
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                if score > 0.8 and class_id < len(self.mobilenet_labels):
                    detected_object = self.mobilenet_labels[class_id]
                    if detected_object == 'cat':
                        self.cat_detections += 1
                        print('Cat detected!')
                    elif detected_object == 'dog':
                        self.dog_detections += 1
                        print('Dog detected!')

    def process_accumulated_detections(self):
        if self.total_detections == 0:
            self.direction = 'forward'
            self.state = 'start_action'
            return

        detection_features = np.zeros(5)  # Adjusted to 5 features
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
        lidar_data_sequence = np.tile(self.lidar_data, (SEQ_LENGTH, 1))
        action_history = self.get_action_history().reshape(SEQ_LENGTH, -1)
        input_features = np.concatenate((detection_sequence, lidar_data_sequence, action_history), axis=1)
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
        forward_command.linear.x = 0.5  # There should be no twist for forward
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
        print(f"Model output: {output}")
        
        # Create a list of (action_index, score) tuples
        action_scores = [(i, output[0][i].item()) for i in range(len(ACTIONS))]
        
        # Sort actions by score in descending order
        sorted_actions = sorted(action_scores, key=lambda x: x[1], reverse=True)
        
        chosen_action = None
        for action_index, score in sorted_actions:
            action = ACTIONS[action_index]
            if self.is_move_possible(action):
                chosen_action = action_index
                self.direction = ['forward', 'backward', 'left', 'right'][action_index]
                self.command = self.action_to_twist(action)
                print(f"Chosen action: {self.direction} (index: {action_index}, score: {score})")
                break
        
        if chosen_action is None:
            print("No possible moves. Staying in place.")
            chosen_action = -1  # Indicate no move
            self.direction = 'stay'
            self.command = Twist()  # Empty twist command (no movement)
        
        # Update action history
        new_action = np.zeros(len(ACTIONS))
        if chosen_action != -1:
            new_action[chosen_action] = 1.0
        self.action_history.append(new_action)
        self.last_movement_direction = chosen_action  # Update last movement direction

    def is_move_possible(self, action):
        print(f"Checking if move is possible. Action: {action}, Filtered LiDAR data: {self.lidar_data}")
        if action == (0, -1) and self.lidar_data[0] == 1:  # Forward
            print("Forward move not possible due to obstacle")
            return False
        if action == (0, 1) and self.lidar_data[1] == 1:  # Backward
            print("Backward move not possible due to obstacle")
            return False
        if action == (-1, 0) and self.lidar_data[2] == 1:  # Left
            print("Left turn not possible due to obstacle")
            return False
        if action == (1, 0) and self.lidar_data[3] == 1:  # Right
            print("Right turn not possible due to obstacle")
            return False
        print("Move is possible")
        return True
        
    def action_to_twist(self, action):
        twist = Twist()
        if action == (0, -1):  # Forward
            twist.linear.x = 0.5  # There should be no twist for forward
        elif action == (0, 1):  # Backward
            twist.angular.z = 1.0
        elif action == (-1, 0):  # Left
            twist.angular.z = 1.0
        elif action == (1, 0):  # Right
            twist.angular.z = -1.0
        return twist

    def check_detection_timer_callback(self):
        if self.pause_manager.is_paused():
            return
        current_time = time.time()
        if current_time - self.last_detection_time > 30:
            print('No detection data received for 30 seconds. Still trying to connect...')
            self.waiting_message_printed = False

def listen_for_pause(pause_manager, nn_node):
    import sys
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("Press 'Space' to pause/resume")
        while True:
            key = sys.stdin.read(1)
            if key == ' ':
                nn_node.toggle_pause()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    pause_manager = PauseManager()
    improved_nn_inference_node = ImprovedNNInferenceNode(pause_manager)
    
    # Start the pause listener thread
    pause_listener_thread = threading.Thread(target=listen_for_pause, args=(pause_manager, improved_nn_inference_node), daemon=True)
    pause_listener_thread.start()
    
    try:
        rclpy.spin(improved_nn_inference_node)
    except KeyboardInterrupt:
        pass
    finally:
        improved_nn_inference_node.finalize_videos()
        improved_nn_inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
