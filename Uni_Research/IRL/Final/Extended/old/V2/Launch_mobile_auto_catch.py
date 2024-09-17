import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
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
import math
import queue
import PyKDL # sudo apt-get install python3-pykdl
import threading

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SEQ_LENGTH = 30  # Length of the action and sensor history

class Action:
    def __init__(self, action_type, target_value):
        self.action_type = action_type
        self.target_value = target_value
        self.start_value = None
        self.movement_confirmed = False
        self.total_angle_turned = 0.0  # New attribute

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
        self.action_lock = threading.Lock()
        self.current_action = None
        self.action_stage = 'idle'  # 'idle', 'executing', 'waiting'
        self.action_timer = self.create_timer(0.01, self.action_timer_callback)  # 10 Hz timer for action execution

        self.pause_manager = pause_manager
        self.bridge = CvBridge()
        self.action_queue = deque()
        self.current_action = None
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
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

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
        for _  in range(SEQ_LENGTH):
            self.action_history.append(np.zeros(len(ACTIONS)))

        # State management
        self.state = 'wait_for_message'
        self.command = Twist()
        self.direction = 'unknown'
        self.action_complete = True
        self.create_timer(0.01, self.timer_callback)  # 10 Hz timer for state machine

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

        # Image and Video creation
        self.frame_width = 640
        self.frame_height = 480
        self.frame_rate = 1/5.0  # 1 frame every 5 seconds
        self.image_count = 0
        self.video_count = 0
        self.current_image_folder = None
        self.create_new_image_folder()

         # Queue for image saving
        self.image_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self.save_images_thread, daemon=True)
        self.save_thread.start()

        # Odometry and IMU data
        self.current_position = None
        self.current_orientation = None
        self.current_angular_velocity = None

        self.initial_waiting_period = 40.0  # Initial waiting period of 40 seconds
        self.stopping_time_factor = 1/8 # spend this proportion of this waiting time stopping at the begging
        self.stop_period = 10.0  # 5 seconds stop period
        self.stop_signal_interval = 0.1  
        self.waiting_start_time = None
        self.stop_start_time = None
        self.detections_during_wait = []
        self.command_sent = False
        self.waiting_for_detection = False  # Flag to indicate if we're waiting for a detection after the initial period
        self.action_start_time = None
        self.action_timeout = 30.0  # 10 seconds timeout for actions
        self.last_cmd_vel_time = None
        self.cmd_vel_timeout = 1.5  # 500ms timeout for cmd_vel messages
        self.yaw_history = deque(maxlen=5)  # Store last 5 yaw readings
        
        self.largest_realistic_turning_angle = 0.4 # modify if apprpriate, will be effected by refresh rate

        

        stopping_angle = math.pi/2 # 90 degrees
        stopping_distance = 0.1 # 10cm

        # apply adjustment to deal with delayed stopping error
        self.stopping_time_error_angel_adjustment_factor=0.7 # lower corresponds to bigger observed error
        self.stopping_time_error_distance_adjustment_factor=0.25
        self.stopping_angle= self.stopping_time_error_angel_adjustment_factor*stopping_angle
        self.stopping_distance = self.stopping_time_error_distance_adjustment_factor*stopping_distance

        self.cat_centric_mode = True
        self.dog_centric_mode = False
        self.DEBUG= False
        
        print("Node initialized. Press 'Space' to pause/resume.")

    def odom_callback(self, msg):
        if self.current_position is None:
            print("Received first odometry data")
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def imu_callback(self, msg):
        self.current_angular_velocity = msg.angular_velocity

    def print_state(self):
            print(f"Current state: {self.state}, Action queue size: {len(self.action_queue)}, Current action: {self.current_action.action_type if self.current_action else 'None'}")  

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

    def lidar_callback(self, msg):
        if self.pause_manager.is_paused():
            return

        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        current_data = np.zeros(5)

        # Function to get index for a specific angle
        def get_index(angle):
            return int((angle - angle_min) / angle_increment)

        # Check for obstacles in different directions with custom ranges
        left_indices = [get_index(angle) for angle in [-20*math.pi/180, -15*math.pi/180, -10*math.pi/180, 0, 10*math.pi/180, 15*math.pi/180,20*math.pi/180]]
        right_indices = [get_index(angle) for angle in [160*math.pi/180, 170*math.pi/180, 180*math.pi/180, 190*math.pi/180, 200*math.pi/180]]
        back_indices= [get_index(angle) for angle in [70*math.pi/180, 80*math.pi/180, 90*math.pi/180, 100*math.pi/180, 110*math.pi/180]]
        front_indices = [get_index(angle) for angle in [230*math.pi/180, 240*math.pi/180, 250*math.pi/180, 260*math.pi/180, 270*math.pi/180, 280*math.pi/180, 290*math.pi/180, 300*math.pi/180, 310*math.pi/180]]

        front_obstacles = any(0.14 <= ranges[i] < 0.16 for i in front_indices if i < len(ranges))
        back_obstacles = any(0.00 <= ranges[i] < 0.05 for i in back_indices if i < len(ranges))
        left_obstacles = any(0.00 <= ranges[i] < 0.05 for i in left_indices if i < len(ranges))
        right_obstacles = any(0.00 <= ranges[i] < 0.05 for i in right_indices if i < len(ranges))

        if front_obstacles:
            current_data[0] = 1
        if back_obstacles:
            current_data[1] = 1
        if left_obstacles:
            current_data[2] = 1
        if right_obstacles:
            current_data[3] = 1

        self.lidar_data_history.append(current_data)
        self.lidar_data = np.mean(self.lidar_data_history, axis=0) > 0.5

    def print_obstacle_info(self, obstacle_data):
        directions = ['front', 'back', 'left', 'right']
        obstacles = [directions[i] for i, value in enumerate(obstacle_data) if value == 1]

        if obstacles:
            print(f"Obstacles: {', '.join(obstacles)}")
        else:
            print("Obstacles: none")

    def create_new_image_folder(self):
            self.image_count = 0
            self.current_image_folder = f"outputimages{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.current_image_folder, exist_ok=True)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_queue.put(cv_image)

    def save_images_thread(self):
        while True:
            frame = self.image_queue.get()
            if frame is None:
                break
            self.save_image(frame)
            self.image_queue.task_done()

    def save_image(self, frame):
        if frame is None or frame.size == 0 or np.max(frame) == 0:
            print("Warning: Invalid frame, not saving")
            return

        if frame.shape[0] != self.frame_height or frame.shape[1] != self.frame_width:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        filename = os.path.join(self.current_image_folder, f"frame{self.image_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        self.image_count += 1
        print(f"Saved image: {filename}")

    def toggle_pause(self):
        if self.pause_manager.is_paused():
            self.resume_recording()
        else:
            self.pause_recording()
        self.pause_manager.toggle_pause()

    def pause_recording(self):
        # Wait for all images to be saved
        self.image_queue.join()
        self.create_video_from_images()
        print("Recording paused")

    def resume_recording(self):
        self.create_new_image_folder()
        print("Recording resumed")

    def create_video_from_images(self):
        if self.image_count == 0:
            print("No images to create video")
            return

        self.video_count += 1
        video_filename = f"outputvideo{self.video_count:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_filename, fourcc, self.frame_rate, (self.frame_width, self.frame_height))

        for i in range(self.image_count):
            image_path = os.path.join(self.current_image_folder, f"frame{i:05d}.jpg")
            frame = cv2.imread(image_path)
            if frame is not None:
                video.write(frame)

        video.release()
        print(f"Created video: {video_filename}")

    def finalize_videos(self):
        # Signal the save thread to stop
        self.image_queue.put(None)
        self.save_thread.join()
        self.create_video_from_images()
        print("All videos finalized")

    def __del__(self):
        self.finalize_videos()

    def add_action(self, action_type, target_value):
        with self.action_lock:
            if len(self.action_queue) == 0:
                self.action_queue.append(Action(action_type, target_value)) 

    def check_detection_timer_callback(self):
        if self.pause_manager.is_paused():
            return
        current_time = time.time()
        if current_time - self.last_detection_time > 30:
            print('No detection data received for 30 seconds. Still trying to connect...')
            self.waiting_message_printed = False

    def timer_callback(self):
        if self.pause_manager.is_paused():
            if not self.was_paused:
                self.stop_action()
                self.finalize_videos()
                self.was_paused = True
            return
        else:
            if self.was_paused:
                self.create_video_from_images()
            self.was_paused = False

        if self.state == 'wait_for_message':
            if self.total_detections > 0:
                self.state = 'process_action'
                self.print_state()
                if self.DEBUG == True:
                    print("Changing state to process_action")

        elif self.state == 'process_action':
            self.process_accumulated_detections()
            self.state = 'wait_for_message'
            self.print_state()
            if self.DEBUG == True:
                print("Processed detections, changing state to wait_for_message")

    def execute_action(self, action):
        twist = Twist()
        if action.action_type == 'forward':
            twist.linear.x = 0.5
            action.start_value = self.current_position.x if self.current_position else None
        elif action.action_type == 'backward':
            twist.linear.x = -0.5
            action.start_value = self.current_position.x if self.current_position else None
        elif action.action_type in ['left', 'right']:
            twist.angular.z = 0.5 if action.action_type == 'left' else -0.5
            action.start_value = self.get_current_yaw()
            action.total_angle_turned = 0.0  # Reset total angle turned
        elif action.action_type == 'stay':
            pass  # No movement for 'stay'
        else:
            print(f"Unknown action type: {action.action_type}")
            return

        self.publish_cmd_vel(twist)
        if self.DEBUG == True:
            print(f"Executing action: {action.action_type}")

    def continue_action(self, action):
        twist = Twist()
        if action.action_type == 'forward':
            twist.linear.x = 0.5
            action.start_value = self.current_position.x if self.current_position else None
        elif action.action_type == 'backward':
            twist.linear.x = -0.5
            action.start_value = self.current_position.x if self.current_position else None
        elif action.action_type in ['left', 'right']:
            twist.angular.z = 0.5 if action.action_type == 'left' else -0.5
            action.start_value = self.get_current_yaw()
        elif action.action_type == 'stay':
            pass  # No movement for 'stay'
        else:
            print(f"Unknown action type: {action.action_type}")
            return

        self.publish_cmd_vel(twist)
        if self.DEBUG == True:
            print(f"Executing action: {action.action_type}")


    def publish_cmd_vel(self, twist=None):
        if twist is None:
            twist = Twist()  # This will be a stop command
        self.publisher.publish(twist)
        self.last_cmd_vel_time = self.get_clock().now().nanoseconds / 1e9

    def is_action_complete(self, action):
        if action.start_value is None:
            return False
        if action.action_type in ['forward', 'backward']:
            if self.current_position:
                distance_moved = abs(self.current_position.x - action.start_value)
                return distance_moved >= action.target_value
        elif action.action_type in ['left', 'right']:
            current_yaw = self.get_current_yaw()
            if current_yaw is not None:
                angle_turned = abs(self.normalize_angle(current_yaw - action.start_value))
                if angle_turned > self.largest_realistic_turning_angle:
                    angle_turned = self.largest_realistic_turning_angle
                action.total_angle_turned += angle_turned
                action.start_value = current_yaw  # Update start value for next iteration
                if self.DEBUG == True:
                    print(f"Rotation check: start_yaw={action.start_value:.2f}, current_yaw={current_yaw:.2f}, "
                        f"angle_turned={angle_turned:.2f}, total_angle_turned={action.total_angle_turned:.2f}, "
                        f"target={action.target_value:.2f}")
                return action.total_angle_turned >= action.target_value
        elif action.action_type == 'stay':
            return True
        return False

    def get_current_yaw(self):
        if self.current_orientation:
            # Normalize the quaternion
            norm = math.sqrt(self.current_orientation.x**2 + self.current_orientation.y**2 + 
                            self.current_orientation.z**2 + self.current_orientation.w**2)
            if norm == 0:
                return None
            normalized_q = PyKDL.Rotation.Quaternion(
                self.current_orientation.x / norm,
                self.current_orientation.y / norm,
                self.current_orientation.z / norm,
                self.current_orientation.w / norm
            )
            _, _, yaw = normalized_q.GetRPY()
            self.yaw_history.append(yaw)
            return np.mean(self.yaw_history)
        return None

    def action_timer_callback(self):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.action_stage == 'idle':
            if self.action_queue:
                self.current_action = self.action_queue.popleft()
                self.execute_action(self.current_action)
                self.action_stage = 'executing'
                self.action_start_time = current_time
                if self.DEBUG == True:
                    print(f"Started action: {self.current_action.action_type}")
            else:
                self.process_accumulated_detections()

        elif self.action_stage == 'executing':
            if self.current_action:
                if self.is_action_complete(self.current_action) or (current_time - self.action_start_time > self.action_timeout):
                    self.stop_action()
                    self.action_stage = 'waiting'
                    self.waiting_start_time = current_time
                    if self.DEBUG == True:
                        print(f"Completed or timed out action: {self.current_action.action_type}, now waiting")
                else:
                    self.continue_action(self.current_action)
                    if self.DEBUG == True:
                        print(f"Continuing action: {self.current_action.action_type}")
            else:
                print("Error: No current action")
                self.action_stage = 'idle'

        elif self.action_stage == 'waiting':
            if current_time - self.waiting_start_time <= self.initial_waiting_period*self.stopping_time_factor:
                self.stop_action()
            if current_time - self.waiting_start_time == self.initial_waiting_period*0.9:
                self.cat_detections = 0
                self.dog_detections = 0
            if current_time - self.waiting_start_time >= self.initial_waiting_period:
                if self.DEBUG == True:
                    print(f"Finished waiting after {self.current_action.action_type}")
                self.current_action = None
                self.action_start_time = None
                self.action_stage = 'idle'
            else:
                remaining_time = self.initial_waiting_period - (current_time - self.waiting_start_time)
                if self.DEBUG == True:
                    print(f"Still waiting, remaining time: {remaining_time:.2f}")

        # Ensure cmd_vel messages are sent frequently
        if self.last_cmd_vel_time is None or (current_time - self.last_cmd_vel_time > self.cmd_vel_timeout):
            self.publish_cmd_vel()

    def send_action_command(self, action):
        twist = Twist()
        if action.action_type == 'forward':
            twist.linear.x = 0.5
            action.start_value = self.current_position.x if self.current_position else 0
        elif action.action_type == 'backward':
            twist.linear.x = -0.5
            action.start_value = self.current_position.x if self.current_position else 0
        elif action.action_type == 'left' or action.action_type == 'right':
            twist.angular.z = 0.5 if action.action_type == 'left' else -1.0
            if self.current_orientation:
                rotation = PyKDL.Rotation.Quaternion(
                    self.current_orientation.x,
                    self.current_orientation.y,
                    self.current_orientation.z,
                    self.currentorientation.w
                )
                _, _, yaw = rotation.GetRPY()
                action.start_value = yaw
            else:
                action.start_value = 0
        elif action.action_type == 'stay':
            pass  # No movement for 'stay'
        else:
            print(f"Unknown action type: {action.action_type}")
            return

        self.publisher_.publish(twist)
        if self.DEBUG == True:
            print(f"Sent command for action: {action.action_type}")

        action.movement_confirmed = True

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def stop_action(self):
        stop_command = Twist()
        self.publisher.publish(stop_command)
        if self.DEBUG == True:
            print('Stopping action')

    def detection_callback(self, msg):
        if self.pause_manager.is_paused():
            return
        self.last_detection_time = time.time()
        if self.action_stage == 'waiting':
            self.detections_during_wait.extend(msg.detections)
        else:
            self.process_detections(msg)

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
        detection_features = np.zeros(5)

        for detection in self.detections_during_wait:
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                if score > 0.8 and class_id < len(self.mobilenet_labels):
                    detected_object = self.mobilenet_labels[class_id]
                    if detected_object == 'cat':
                        self.cat_detections += 1
                    elif detected_object == 'dog':
                        self.dog_detections += 1

        if self.waiting_start_time is not None and self.initial_waiting_period is not None and time.time() - self.waiting_start_time == self.initial_waiting_period*0.9:
            print(f"Processed {self.total_detections} detections: {self.cat_detections} cats, {self.dog_detections} dogs")

        if len(self.action_queue) == 0:
            
            if self.cat_detections > self.dog_detections:
                if self.last_movement_direction is not None:
                    detection_features[self.last_movement_direction] = 2
                print("Processing cat detection")
            elif self.dog_detections > self.cat_detections:
                if self.last_movement_direction is not None:
                    detection_features[self.last_movement_direction] = 3
                print("Processing dog detection")
            else:
                print("Processing other detection")

            input_tensor = self.create_input_tensor(detection_features)
            action_index = self.process_model_output(input_tensor)

            if action_index is not None:
                direction = ['forward', 'backward', 'left', 'right'][action_index]
                target_value = self.stopping_distance if direction in ['forward', 'backward'] else self.stopping_angle # 10 cm or 90 degrees
                new_action = Action(direction, target_value)
                self.action_queue.append(new_action)
                print(f"Added action: {direction}")

            self.detections_during_wait = []

    def create_input_tensor(self, detection_features):
        detection_sequence = np.tile(detection_features, (SEQ_LENGTH, 1))
        lidar_data_sequence = np.tile(self.lidar_data, (SEQ_LENGTH, 1))
        action_history = self.get_action_history().reshape(SEQ_LENGTH, -1)
        input_features = np.concatenate((detection_sequence, lidar_data_sequence, action_history), axis=1)
        return torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    def turn(self, direction=None):
        if direction is None:
            direction = self.direction
        if direction == 'left':
            self.command.angular.z = 0.5
        elif direction == 'right':
            self.command.angular.z = -0.5
        elif direction == 'backward':
            self.command.angular.z = 0.5  # Turn left for backward (180-degree rotation)
        self.publisher_.publish(self.command)
        print(f'Turning: {direction}')

    def move_forward(self):
        forward_command = Twist()
        forward_command.linear.x = 0.5  # There should be no twist for forward
        self.publisher.publish(forward_command)
        print('Moving forward')

    def get_action_history(self):
        return np.array(list(self.action_history)).flatten()

    def process_model_output(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        print(f"Model output: {output}")

        action_scores = [(i, output[0][i].item()) for i in range(len(ACTIONS))]
        sorted_actions = sorted(action_scores, key=lambda x: x[1], reverse=True)

        for action_index, score in sorted_actions:
            action = ACTIONS[action_index]
            print(f"Checking action: {action}")
            if self.is_move_possible(action):
                return action_index

        print("No possible moves found")
        return None

    def is_move_possible(self, action):
        print(f"Checking if move is possible. Action: {action}")
        self.print_obstacle_info(self.lidar_data)

        # Check for cat or dog detections
        if self.cat_centric_mode == True:
            cat_detected = self.cat_detections > 0 
            dog_detected = self.dog_detections > 0 and self.cat_detections == 0
        elif self.dog_centric_mode == True:
            dog_detected = self.dog_detections > 0 
            cat_detected = self.cat_detections > 0 and self.dog_detections == 0
        else:
            cat_detected = self.cat_detections > 0 and self.dog_detections  == 0
            dog_detected = self.dog_detections > 0 and self.cat_detections == 0


        if cat_detected:
            # Allow forward movement when a cat is detected
            if action == (0, -1):  # Forward
                print("Forward move possible due to cat detection")
                return True
            else:
                print("Only forward move allowed due to cat detection")
                return False

        elif dog_detected:
            # Allow any movement except forward when a dog is detected
            if action == (0, -1):  # Forward
                print("Forward move not possible due to dog detection")
                return False
            else:
                print("Move possible due to dog detection")
                return True

        # If no cat or dog detected, check for obstacles
        if action == (0, -1) and self.lidar_data[0]:  # Forward and front obstacle
            print("Forward move not possible due to obstacle")
            return False
        elif action == (0, 1) and self.lidar_data[1]:  # Backward and back obstacle
            print("Backward move not possible due to obstacle")
            return False
        elif action == (-1, 0) and self.lidar_data[2]:  # Left and left obstacle
            print("Left turn not possible due to obstacle")
            return False
        elif action == (1, 0) and self.lidar_data[3]:  # Right and right obstacle
            print("Right turn not possible due to obstacle")
            return False

        print("Move is possible")
        return True

    def action_to_twist(self, action):
        twist = Twist()
        if action == (0, -1):  # Forward
            twist.linear.x = 0.5  # There should be no twist for forward
        elif action == (0, 1):  # Backward
            twist.angular.z = 0.5
        elif action == (-1, 0):  # Left
            twist.angular.z = 0.5
        elif action == (1, 0):  # Right
            twist.angular.z = -0.5
        return twist

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

