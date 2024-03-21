import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import FollowWaypoints
import yaml
import argparse
from PIL import Image, ImageOps
import os
import subprocess
import numpy as np
from scipy.ndimage import label
import math
import time

class WaypointNavigator(Node):
    def __init__(self, map_yaml_path):
        super().__init__('waypoint_navigator')
        self.client = ActionClient(self, FollowWaypoints, '/follow_waypoints', callback_group=ReentrantCallbackGroup())
        self.current_position = None
        self.waypoint_timer = None
        self.waypoint_timeout = 25  # seconds
        self.last_waypoint_time = None

        self.create_subscription(Odometry, 'odom', self.odom_callback, QoSProfile(depth=10))
        self.waypoints = self.generate_waypoints(map_yaml_path, 50)  # Ensure this is included and correctly defined
        self.monitor_ros_topics()

    def odom_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.waypoints and not self.waypoint_timer:
            self.send_nearest_waypoint()

    def generate_waypoints(self, map_yaml_path, num_waypoints):
        map_data = self.load_map_metadata(map_yaml_path)
        image_path = os.path.join(os.path.dirname(map_yaml_path), map_data['image'])
        occupancy_grid = ImageOps.grayscale(Image.open(image_path))
        navigable_map = np.array(occupancy_grid) == 0

        structure = np.ones((3, 3), dtype=np.int32)
        labeled, ncomponents = label(navigable_map, structure)

        component_sizes = np.bincount(labeled.ravel())
        largest_component = component_sizes[1:].argmax() + 1
        largest_area_mask = labeled == largest_component

        return self.place_waypoints_in_largest_area(largest_area_mask, map_data['resolution'], map_data['origin'], num_waypoints)

    def place_waypoints_in_largest_area(self, navigable_area, resolution, origin, num_waypoints):
        indices = np.argwhere(navigable_area)
        np.random.shuffle(indices)
        waypoints = []
        for idx in indices[:num_waypoints]:
            global_x = origin[0] + idx[1] * resolution
            global_y = origin[1] + idx[0] * resolution
            waypoints.append((global_x, global_y))
        return waypoints

    def load_map_metadata(self, map_yaml_path):
        with open(map_yaml_path, 'r') as file:
            map_data = yaml.safe_load(file)
        if not all(key in map_data for key in ['resolution', 'origin', 'image']):
            raise KeyError("Map metadata must include 'resolution', 'origin', and 'image'")
        return map_data

    def send_nearest_waypoint(self):
        if self.current_position and self.waypoints:
            nearest_waypoint = min(self.waypoints, key=lambda wp: self.distance(self.current_position, wp))
            self.waypoints.remove(nearest_waypoint)
            self.send_waypoint(nearest_waypoint)
            self.waypoint_timer = self.create_timer(self.waypoint_timeout, self.handle_waypoint_timeout)
            self.last_waypoint_time = time.time()

    def send_waypoint(self, waypoint):
        goal_msg = FollowWaypoints.Goal()
        pose = self.create_pose(waypoint)
        goal_msg.poses = [pose]
        self.client.wait_for_server()
        self.future = self.client.send_goal_async(goal_msg)
        self.future.add_done_callback(self.goal_response_callback)

    def handle_waypoint_timeout(self):
        if time.time() - self.last_waypoint_time >= self.waypoint_timeout:
            print(f"Waypoint timeout: Skipping waypoint after {self.waypoint_timeout} seconds")
            self.add_new_waypoint()
            self.send_nearest_waypoint()

    def add_new_waypoint(self):
        if self.waypoints:
            last_waypoint = self.waypoints[-1]
        else:
            last_waypoint = self.current_position
        new_waypoint = (last_waypoint[0] + 1.0, last_waypoint[1] + 1.0)
        self.waypoints.append(new_waypoint)
        print(f"Added new waypoint: ({new_waypoint[0]}, {new_waypoint[1]})")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_result_future = goal_handle.get_result_async()
            self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result is not None:
            print("Waypoint reached successfully.")
            self.send_nearest_waypoint()
        else:
            print("Failed to reach waypoint, retrying...")
            self.send_nearest_waypoint()

    def create_pose(self, waypoint):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = waypoint[0]
        pose.pose.position.y = waypoint[1]
        pose.pose.orientation.w = 1.0
        return pose

    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def monitor_ros_topics(self):
        topics = ["/follow_waypoints/feedback", "/waypoint_navigator/waypoints", "/plan", "/local_plan", "/cmd_vel"]
        for topic in topics:
            self.open_terminal_and_monitor_topic(topic)

    def open_terminal_and_monitor_topic(self, topic):
        command = f"""
        gnome-terminal -- bash -c '
            echo "Monitoring {topic}...";
            ros2 topic echo {topic} || echo "Failed to monitor {topic}.";
            read -p "Press enter to exit...";
            exec bash'
        """
        subprocess.Popen(command, shell=True)

if __name__ == '__main__':
    rclpy.init()
    parser = argparse.ArgumentParser(description='Navigate a robot through a grid of waypoints on a map.')
    parser.add_argument('map_yaml_path', type=str, help='The path to the map.yaml file')
    args = parser.parse_args()
    navigator = WaypointNavigator(args.map_yaml_path)
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()
