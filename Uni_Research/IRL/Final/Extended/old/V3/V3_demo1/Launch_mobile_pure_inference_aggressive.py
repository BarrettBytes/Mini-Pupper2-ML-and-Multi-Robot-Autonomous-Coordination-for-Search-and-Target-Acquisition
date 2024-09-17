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
import os
import time
from collections import deque
import threading
from datetime import datetime
import math
import queue
import PyKDL
from Demo3 import DualNetworkPolicy 

'''
Intended Behavior
The robot control system should follow a clear, cyclical pattern of operation:

Wait for Detections: The robot should start in a waiting state, actively listening for detections from its sensors.
Process Detections: When a detection occurs (cat, dog, or other), the system should immediately process this information.
Consult Neural Network: For every significant detection, the system must consult the NN. This is a critical step that should never be skipped, as it's the core of the robot's decision-making process.
Create Action Based on NN Output: The NN's output should always lead to the creation of an action, even if that action is to remain stationary.
Check Action Feasibility: Before executing the action, the system should verify if the action is possible given the current environmental constraints (e.g., obstacles).
Execute Action: If the action is feasible, add it to the queue and execute it. The system should then wait for the action to complete.
Action Completion: Once the action is complete, the system should loop back to the waiting state, ready for the next detection.
'''

ACTIONS = ['turn_left', 'turn_right', 'move_forward']
SEQ_LENGTH = 30  # Length of the action and sensor history

class DetectionHistory:
    def __init__(self, window_size=1):
        self.window_size = window_size
        self.detections = deque(maxlen=window_size)

    def add_detection(self, object_type):
        self.detections.append((object_type, time.time()))

    def get_weighted_score(self):
        cat_score = 0
        dog_score = 0
        current_time = time.time()
        for obj, timestamp in self.detections:
            weight = 1 - (current_time - timestamp) / (self.window_size * 10)  # 1 seconds window
            if weight < 0:
                weight = 0
            if obj == 'cat':
                cat_score += weight
            elif obj == 'dog':
                dog_score += weight
            print(f"cat score: {cat_score}, dog_score: {dog_score}")
        return cat_score, dog_score
    
    def clear(self):
        self.detections.clear()
    
class Action:
    def __init__(self, action_type, target_value):
        self.action_type = action_type
        self.target_value = target_value
        self.start_value = None
        self.movement_confirmed = False
        self.total_angle_turned = 0.0  # New attribute

class RateLimiter:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.last_called = 0

    def can_process(self):
        current_time = time.time()
        if current_time - self.last_called >= self.min_interval:
            self.last_called = current_time
            return True
        return False

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
        self.action_stage = 'initial_waiting'  # 'idle', 'executing', 'waiting'

        self.pause_manager = pause_manager
        self.bridge = CvBridge()
        self.action_queue = deque()
        self.current_action = None
        self.detection_history = DetectionHistory()

        self.detection_rate_limiter = RateLimiter(0.01)  # 10ms interval
        self.lidar_rate_limiter = RateLimiter(0.01)
        self.image_rate_limiter = RateLimiter(0.01)


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
        self.input_size = 15  # Ensuring the total input size matches the original model
        hidden_size = 64
        num_actions = len(ACTIONS)

        # Initialize and load the trained model
        self.model = DualNetworkPolicy(self.input_size, hidden_size, num_actions)
        # Load the model on CPU
        self.model.load_state_dict(torch.load('0.9_policy.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Initialize action history
        self.action_history = deque(maxlen=SEQ_LENGTH)
        for _  in range(SEQ_LENGTH):
            self.action_history.append(np.zeros(len(ACTIONS)))

        # State management
        self.state = 'initial_waiting'
        self.waiting_start_time = self.get_clock().now().nanoseconds / 1e9
        self.initial_waiting_period = 15  # 15 seconds initial waiting period
        self.set_waiting_period = 5.0  # 5 seconds waiting period

        self.command = Twist()
        self.direction = 'unknown'
        self.action_complete = True

        self.start_time = time.time()
        self.last_detection_time = time.time()

        # Load PASCAL VOC labels
        self.mobilenet_labels = self.load_mobilenet_labels()

        # Timer for connection checking
        self.waiting_message_printed = False

        self.create_timer(0.01, self.action_timer_callback)  # 100 Hz timer for action execution
        self.create_timer(0.1, self.timer_callback)  # 10 Hz timer for state machine
        self.create_timer(30.0, self.check_detection_timer_callback)  # Timer for connection checking

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
        self.frame_rate = 1 # 5 frames every second
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

        self.stopping_time_factor = 1/6 # spend this proportion of this waiting time stopping at the begging
        self.stop_period = 1.0  # 1 seconds stop period
        self.stop_signal_interval = 0.1  
        self.stop_start_time = None
        self.detections_during_wait = []
        self.command_sent = False
        self.waiting_for_detection = False  # Flag to indicate if we're waiting for a detection after the initial period
        self.action_start_time = None
        self.action_timeout = 30.0  # 30 seconds timeout for actions
        self.last_cmd_vel_time = None
        self.cmd_vel_timeout = 0.5  # 500ms timeout for cmd_vel messages
        self.yaw_history = deque(maxlen=5)  # Store last 5 yaw readings
        self.first_camera_message_received = False

        self.largest_realistic_turning_angle = 0.8 # modify if apprpriate, will be effected by refresh rate

        stopping_angle = math.pi/2 # 90 degrees
        stopping_distance = 0.2 # 10cm

        # apply adjustment to deal with delayed stopping error
        self.stopping_time_error_angel_adjustment_factor=0.7 # lower corresponds to bigger observed error
        self.stopping_time_error_distance_adjustment_factor=0.05 # something is definitely wrong with the 10cm distance calculation
        self.stopping_angle= self.stopping_time_error_angel_adjustment_factor*stopping_angle
        self.stopping_distance = self.stopping_time_error_distance_adjustment_factor*stopping_distance
        self.angular_speed = 0.7 # adjust to change turn spped
        self.linear_speed = 0.2 # adjust to change move forward speed

        self.DEBUG= False
        
        print("Node initialized. Press 'Space' to pause/resume.")

    
    def odom_callback(self, msg):
        new_position = msg.pose.pose.position
        new_orientation = msg.pose.pose.orientation
        
        if self.current_position is None:
            print("Received first odometry data")
            self.current_position = new_position
            self.current_orientation = new_orientation
        else:
            if (new_position.x != self.current_position.x or 
                new_position.y != self.current_position.y or 
                new_orientation.z != self.current_orientation.z):
                if self.DEBUG:
                    print("Odometry change detected!")
            
            self.current_position = new_position
            self.current_orientation = new_orientation
        
        if self.DEBUG == True:
            print(f"Odometry update: position=({self.current_position.x:.4f}, {self.current_position.y:.4f}, {self.current_position.z:.4f}), "
                f"orientation=({self.current_orientation.x:.4f}, {self.current_orientation.y:.4f}, {self.current_orientation.z:.4f}, {self.current_orientation.w:.4f})")

    def imu_callback(self, msg):
        self.current_angular_velocity = msg.angular_velocity

    def print_state(self):
            print(f"Current state: {self.state}, Action queue size: {len(self.action_queue)}, Current action: {self.current_action.action_type if self.current_action else 'None'}")  

    def load_mobilenet_labels(self):
        return [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
        ]

    def lidar_callback(self, msg):
        if self.pause_manager.is_paused() or not self.lidar_rate_limiter.can_process():
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

        front_obstacles = any(0.16 <= ranges[i] < 0.20 for i in front_indices if i < len(ranges))
        # the below are only for preventing turns so must be right next to for safety
        # if detecting above essential levels for robot safety then reduce range
        #  (as we dont want to prevent impacting cat)
        back_obstacles = any(0.00 <= ranges[i] < 0.05 for i in back_indices if i < len(ranges))
        left_obstacles = any(0.00 <= ranges[i] < 0.05 for i in left_indices if i < len(ranges))
        right_obstacles = any(0.00 <= ranges[i] < 0.05 for i in right_indices if i < len(ranges))

        if front_obstacles:
            current_data[0] = 1
            print("obstacle front")
            print(f"action: {self.current_action.action_type}")
            print(f"cat detections {self.cat_detections}")
            print(f"dog detections {self.cat_detections}")
            if self.cat_detections >= self.dog_detections and self.current_action.action_type == "move_forward":
                self.stop_action()
                self.state = 'waiting'
                print("stopping")
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

    def save_cat_detection_image(self):
        if not hasattr(self, 'cat_detection_folder'):
            self.create_cat_detection_folder()

        # Get the latest image from the queue
        try:
            latest_image = self.image_queue.get(block=False)
            self.image_queue.put(latest_image)  # Put it back in the queue
            
            # Save the image
            filename = os.path.join(self.cat_detection_folder, f"cat_detected_{self.cat_detections:05d}.jpg")
            cv2.imwrite(filename, latest_image)
            print(f"Saved cat detection image: {filename}")
        except queue.Empty:
            print("No image available to save for cat detection")

    def create_cat_detection_folder(self):
        self.cat_detection_folder = f"cat_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.cat_detection_folder, exist_ok=True)
        print(f"Created folder for cat detections: {self.cat_detection_folder}")

    def create_new_image_folder(self):
            self.image_count = 0
            self.current_image_folder = f"outputimages{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.current_image_folder, exist_ok=True)

    def image_callback(self, msg):
        if self.pause_manager.is_paused() or not self.image_rate_limiter.can_process():
            return
        if self.DEBUG:
            print("can process")
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
        if self.state != 'waiting':
            if frame is None or frame.size == 0 or np.max(frame) == 0:
                print("Warning: Invalid frame, not saving")
                return

            if frame.shape[0] != self.frame_height or frame.shape[1] != self.frame_width:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            # Add text to the image
            direction_text = f"Direction: {self.get_current_direction()}"
            detection_text = "No detection"
            if self.cat_detections > 0:
                detection_text = "Cat detected"
            elif self.dog_detections > 0:
                detection_text = "Dog detected"
            obstacle_text = self.get_obstacle_info()
            
            cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, obstacle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            filename = os.path.join(self.current_image_folder, f"frame{self.image_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            self.image_count += 1
            print(f"Saved image: {filename}")

    def get_obstacle_info(self):
        directions = ['front', 'back', 'left', 'right']
        obstacles = [directions[i] for i, value in enumerate(self.lidar_data[:4]) if value == 1]
        if obstacles:
            return f"Obstacles: {', '.join(obstacles)}"
        return "No obstacles"

    def get_current_direction(self):
        if self.current_action:
            return self.current_action.action_type
        return "idle"

    def toggle_pause(self):
        if self.pause_manager.is_paused():
            self.resume_recording()
            self.resume_operation()
        else:
            self.pause_recording()
            self.pause_operation()
        self.pause_manager.toggle_pause()

    def pause_operation(self):
        self.stop_action()  # Send stop command to the robot
        print("Operation paused")

    def resume_operation(self):
        print("Operation resumed")
        # If there was an action in progress when paused, resume it
        if self.current_action:
            self.action_stage = 'executing'
            self.execute_action(self.current_action)
        else:
            self.action_stage = 'idle'

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
        video_filename = f"outputvideo{self.video_count:03d}{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
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
        try:
            self.finalize_videos()
        except AttributeError:
            print("Warning: Unable to finalize videos. Some attributes might not have been initialized.")

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

        if self.state == 'waiting':
            if self.total_detections > 0:
                self.state = 'process_action'
                self.print_state()
                if self.DEBUG:
                    print("Changing state to process_action")

        elif self.state == 'process_action':
            self.begin_execution_phase()
            self.print_state()
            if self.DEBUG:
                print("Processed detections, changing state to wait_for_message")

    def execute_action(self, action):
        twist = Twist()
        if action.action_type == 'move_forward':
            twist.linear.x = self.linear_speed
        elif action.action_type == 'turn_left':
            twist.angular.z = self.angular_speed
        elif action.action_type == 'turn_right':
            twist.angular.z = -self.angular_speed
        else:
            print(f"Unknown action type: {action.action_type}")
            return

        self.publish_cmd_vel(twist)
        print(f"Executing action: {action.action_type} with twist: linear={twist.linear.x}, angular={twist.angular.z}")

    def continue_action(self, action):
        twist = Twist()
        if action.action_type == 'move_forward':
            twist.linear.x = self.linear_speed
        elif action.action_type == 'turn_left':
            twist.angular.z = self.angular_speed
        elif action.action_type == 'turn_right':
            twist.angular.z = - self.angular_speed
        else:
            print(f"Unknown action type: {action.action_type}")
            return

        self.publish_cmd_vel(twist)
        if self.DEBUG:
            print(f"Continuing action: {action.action_type} with twist: linear={twist.linear.x}, angular={twist.angular.z}")
   

    def publish_cmd_vel(self, twist=None):
        if twist is None:
            twist = Twist()  # This will be a stop command
        self.publisher.publish(twist)
        self.last_cmd_vel_time = self.get_clock().now().nanoseconds / 1e9
        if self.DEBUG:
            print(f"Published cmd_vel: linear={twist.linear.x}, angular={twist.angular.z}")

    def is_action_complete(self, action):
        if action.start_value is None:
            return False
        if action.action_type == 'move_forward':
            if self.current_position and action.start_position:
                distance_moved = math.sqrt(
                    (self.current_position.x - action.start_position.x)**2 +
                    (self.current_position.y - action.start_position.y)**2 +
                    (self.current_position.z - action.start_position.z)**2
                )
                return distance_moved >= action.target_value
        elif action.action_type in ['turn_left', 'turn_right']:
            current_yaw = self.get_current_yaw()
            if current_yaw is not None:
                angle_turned = abs(self.normalize_angle(current_yaw - action.start_value))
                if angle_turned > self.largest_realistic_turning_angle:
                    angle_turned = self.largest_realistic_turning_angle
                action.total_angle_turned += angle_turned
                action.start_value = current_yaw  # Update start value for next iteration
                if self.DEBUG:
                    print(f"Rotation check: start_yaw={action.start_value:.2f}, current_yaw={current_yaw:.2f}, "
                        f"angle_turned={angle_turned:.2f}, total_angle_turned={action.total_angle_turned:.2f}, "
                        f"target={action.target_value:.2f}")
                return action.total_angle_turned >= action.target_value
        return False

    def get_current_yaw(self):
        if self.current_orientation:
            # Normalize the quaternion
            norm = math.sqrt(self.current_orientation.x**2 + self.current_orientation.y**2 + 
                            self.current_orientation.z**2 + self.current_orientation.w**2)
            if norm == 0:
                print("Warning: Zero norm in quaternion")
                return None
            normalized_q = PyKDL.Rotation.Quaternion(
                self.current_orientation.x / norm,
                self.current_orientation.y / norm,
                self.current_orientation.z / norm,
                self.current_orientation.w / norm
            )
            _, _, yaw = normalized_q.GetRPY()
            self.yaw_history.append(yaw)
            mean_yaw = np.mean(self.yaw_history)
            print(f"Current yaw: {yaw}, Mean yaw: {mean_yaw}")
            return mean_yaw
        else:
            print("Warning: No current orientation data")
            return None

    
    
    def action_timer_callback(self):
        if self.pause_manager.is_paused():
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.waiting_start_time

        if self.DEBUG == True:
            print(f"Action state callback: {self.state}, Elapsed time: {elapsed_time:.2f}")

        if self.state == 'initial_waiting':
            if elapsed_time >= self.initial_waiting_period:
                print("Initial waiting period over, moving to idle state")
                self.state = 'idle'
        elif self.state == 'idle':
            self.begin_execution_phase()
        elif self.state == 'executing':
            if self.current_action:
                if self.is_action_complete(self.current_action) or (current_time - self.action_start_time > self.action_timeout):
                    self.stop_action()
                    self.state = 'waiting'
                    self.waiting_start_time = current_time
                    print(f"Action completed or timed out: {self.current_action.action_type}")
                else:
                    self.continue_action(self.current_action)
            else:
                print("Error: No current action in executing state")
                self.state = 'idle'
        elif self.state == 'waiting':
            if elapsed_time >= self.set_waiting_period:
                self.state = 'idle'
                print("Waiting period over, moving to idle state")

        self.publish_cmd_vel()
        

    def begin_execution_phase(self):
        print("Beginning execution phase")
        cat_score, dog_score = self.process_accumulated_detections()
        action_index = self.consult_NN(cat_score, dog_score)
        if action_index is not None:
            action_type = ACTIONS[action_index]
            target_value = self.stopping_distance if action_type == 'move_forward' else self.stopping_angle
            self.current_action = Action(action_type, target_value)
            self.execute_action(self.current_action)
            self.state = 'executing'
            self.action_start_time = self.get_clock().now().nanoseconds / 1e9
            print(f"Starting new action: {action_type}")
            self.clear_detection_history()
        else:
            print("No valid action from NN, staying in idle state")
            self.state = 'idle'
        print(f"Current state after begin_execution_phase: {self.state}")


    def clear_detection_history(self):
        self.cat_detections = 0
        self.dog_detections = 0
        self.detection_history.clear()
        
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def stop_action(self):
        stop_command = Twist()
        self.publisher.publish(stop_command)
        if self.DEBUG:
            print('Stopping action')

    def detection_callback(self, msg):
        if self.pause_manager.is_paused() or not self.detection_rate_limiter.can_process():
            return
        if self.DEBUG:
            print("can process detection")
        self.first_camera_message_received = True
        self.last_detection_time = time.time()
        if self.action_stage == 'waiting':
            self.detections_during_wait.extend(msg.detections)
        elif self.action_stage == 'initial_waiting':
            self.process_detections(msg)
        else:
            self.process_detections(msg)

    def process_detections(self, detections_msg):
        processing_detections = self.detections_during_wait
        processing_detections.extend(detections_msg.detections)
        for detection in processing_detections:
                for result in detection.results:
                    class_id = int(result.hypothesis.class_id)
                    score = result.hypothesis.score
                    if score > 0.8 and class_id < len(self.mobilenet_labels):
                        detected_object = self.mobilenet_labels[class_id]
                        if detected_object == 'cat':
                            self.cat_detections += 1
                            self.detection_history.add_detection('cat')
                            if self.state in ['idle', 'waiting']:
                                self.begin_execution_phase()
                        elif detected_object == 'dog':
                            self.dog_detections += 1
                            self.detection_history.add_detection('dog')
                            if self.state in ['idle', 'waiting']:
                                self.begin_execution_phase()
        self.detections_during_wait.clear()

    def interrupt_wait_phase(self):
        if self.state == 'waiting':
            self.state = 'idle'
            self.begin_execution_phase()
    
    def process_accumulated_detections(self):
        print("processing accumilated actions")
        cat_score, dog_score = self.detection_history.get_weighted_score()
        if cat_score > dog_score:
            print("Processing cat detection")
            # Logic for cat detection
        elif dog_score > cat_score:
            print("Processing dog detection")
            # Logic for dog detection
        else:
            print("No significant detection")
        return cat_score, dog_score

    def create_input_tensor(self, detection_features):
        # Ensure detection_features has 4 elements (camera data)
        detection_features = np.pad(detection_features, (0, max(0, 4 - len(detection_features))))[:4]
        
        # Create sequences
        detection_sequence = np.tile(detection_features, (SEQ_LENGTH, 1))  # Shape: (SEQ_LENGTH, 4)
        lidar_data_sequence = np.tile(self.lidar_data[:5], (SEQ_LENGTH, 1))  # Shape: (SEQ_LENGTH, 5)
        action_history = np.array(list(self.action_history)).reshape(SEQ_LENGTH, -1)  # Shape: (SEQ_LENGTH, 3)
        
        # Combine all features
        input_features = np.concatenate((
            detection_sequence,  # 4 features
            lidar_data_sequence,  # 5 features
            action_history,  # 3 features
            np.zeros((SEQ_LENGTH, 3))  # 3 additional features to reach 15
        ), axis=1)
        
        # Ensure the input size matches the expected size
        assert input_features.shape[1] == self.input_size, f"Input size mismatch. Expected {self.input_size}, got {input_features.shape[1]}"
        
        return torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


    def get_action_history(self):
        return np.array(list(self.action_history)).flatten()

    def get_detection_features(self, cat_score, dog_score):
        detection_features = np.zeros(5)
        if cat_score > dog_score:
            detection_features[self.last_movement_direction] = 2  # Assuming 2 represents cat detection
        elif dog_score > cat_score:
            detection_features[self.last_movement_direction] = 3  # Assuming 3 represents dog detection
        return detection_features

    def consult_NN(self, cat_score, dog_score):
        input_tensor = self.create_input_tensor(self.get_detection_features(cat_score, dog_score))
        action_index = self.process_model_output(input_tensor, cat_score, dog_score)
        return action_index

    def process_model_output(self, input_tensor, cat_score, dog_score):
        with torch.no_grad():
            action_values = self.model(input_tensor)
        print(f"Model output: {action_values}")

        sorted_actions = torch.argsort(action_values[0], descending=True)
        
        # Create camera data based on detections
        camera_data = np.zeros(4)
        if cat_score > dog_score:
            camera_data[0] = 2  # Cat detected in front
        elif dog_score > cat_score:
            camera_data[0] = 3  # Dog detected in front

        for action_index in sorted_actions:
            action = ACTIONS[action_index.item()]
            print(f"Checking action: {action}")
            
            if self.is_move_possible(action, cat_score, dog_score):
                action_encoding = [0, 0, 0]
                action_encoding[action_index.item()] = 1
                
                input_data = torch.tensor(np.concatenate([camera_data, action_encoding]), dtype=torch.float32).unsqueeze(0)
                safety_score = self.model.short_term_context_network(input_data).item()
                
                print(f"Safety score for {action}: {safety_score}")
                
                if safety_score >= self.model.safety_threshold:
                    print(f"Selected safe action: {action}")
                    return action_index.item()
                else:
                    print(f"Action {action} deemed unsafe")

        print("No safe actions found")
        return None
    
    # As the decisions are driven by the NeuralNet we don't want to hard code much here
    # Dog avoidance is not hard coded
    # obstacle avoidance is however as it is essential for robot safety
    # if there is a cat forward then the obstace can be ignored as it is the cat
    def is_move_possible(self, action, cat_score, dog_score):
        print(f"Checking if move is possible. Action: {action}")
        self.print_obstacle_info(self.lidar_data)
        cat_detected = cat_score > dog_score

        # Allow forward movement when a cat is detected
        if cat_detected:
            if action == 'move_forward':
                print("Forward move possible due to cat detection")
                return True
          

        # If no cat detected, check for obstacles
        if action == 'move_forward' and self.lidar_data[0]:
            print("Forward move not possible due to obstacle")
            return False
        elif action == 'turn_left':
            if self.lidar_data[2]:
                return False
            return True
        elif action == 'turn_right':
            if self.lidar_data[3]:
                return False
            return True

        print("Move is possible")
        return True

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
        while rclpy.ok():
            rclpy.spin_once(improved_nn_inference_node, timeout_sec=0.1)
            if not pause_manager.is_paused():
                improved_nn_inference_node.timer_callback()
    except KeyboardInterrupt:
        pass
    finally:
        improved_nn_inference_node.finalize_videos()
        improved_nn_inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()