import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import FollowWaypoints
from std_msgs.msg import Bool
import yaml
import argparse
from PIL import Image, ImageOps
import os
import subprocess
import numpy as np
from scipy.ndimage import label
import math
import time

class Waypoint_Sender_Node(Node):
    def __init__(self, map_yaml_path):
        super().__init__('Waypoint_Sender_Node')
        # create a client who can send goals to follow_waypoints action server
        self.waypoint_client = ActionClient(self, FollowWaypoints, '/follow_waypoints', callback_group=ReentrantCallbackGroup())
        self.pupper_position = None
        self.patrol_status = True

        # subscribe to bottle_detected from bottle finder script pupper can stop if bottle found
        self.create_subscription(Bool, 'bottle_detected', self.bottle_detected_callback, 10)
        # subscribe to odom for current position
        self.create_subscription(Odometry, 'odom', self.odom_callback, QoSProfile(depth=10))
        # generate waypoints
        self.generated_waypoints = self.create_waypoints(map_yaml_path, 50)
        # monitor topic info from ROS
        self.open_windows_to_monitor_ROS_topics()

    def odom_callback(self, message):
        # get current position from odom
        self.pupper_position = (message.pose.pose.position.x, message.pose.pose.position.y)
        # if patrol is active and list of created waypoints isn't empty
        if self.generated_waypoints and self.patrol_status:
            # send the next closest waypoint
            self.send_next_closest_waypoint()

    def bottle_detected_callback(self, message):
        # if the bottle is detected stop the patrol and display current co-ordinates
        if message.data:
            self.get_logger().info(f'Bottle was detected at: {self.pupper_position}, pupper patrol stopping')
            self.patrol_status = False

    # generate the waypoints
    def create_waypoints(self, path_to_map_yaml, waypoint_count):
        # load the map metadata from the generated yaml map
        map_metadata = self.load_map(path_to_map_yaml)
        # load image of the map
        path_to_image = os.path.join(os.path.dirname(path_to_map_yaml), map_metadata['image'])
        # from the image, the light parts are the navigatable space, get these to form the occupancy grid
        Occupiable_space = ImageOps.grayscale(Image.open(path_to_image))
        # form the navigatable space from this grid
        navigable_area_map = np.array(Occupiable_space) == 0

        # identify elements that are connected with an equal value
        connected_elements, _ = label(navigable_area_map, np.ones((3, 3), dtype=np.int32))

        # find size of connected areas
        connected_area_sizes = np.bincount(connected_elements.ravel())
        # find and mask largest area
        largest_area = connected_area_sizes[1:].argmax() + 1
        largest_area_mask = connected_elements == largest_area
        # place the waypoints in this area
        return self.place_waypoints(largest_area_mask, map_metadata['resolution'], map_metadata['origin'], waypoint_count)

    def load_map(self, map_yaml_path):
        with open(map_yaml_path, 'r') as file:
            map_data = yaml.safe_load(file)
        if not all(key in map_data for key in ['resolution', 'origin', 'image']):
            raise KeyError("Map metadata must include 'resolution', 'origin', and 'image'")
        return map_data

    def place_waypoints(self, navigable_map, map_resolution, map_origin, waypoint_count):
        # get index's from navigatable area
        navigatable_points = np.argwhere(navigable_map)
        # randomise index's (patrol will be random with this line so may cut in future)
        # RL algorithm could be used here to optimise
        # Although RL might be best if solution is found using only live data and no script
        np.random.shuffle(navigatable_points)

        # setup waypoints
        waypoints = []
        for navigatable_point in navigatable_points[:waypoint_count]:
            # get the x and y co-ordiantes by adding origin to navigatable
            # point from the navigatable_map mask made earlier multiplied
            # by the resolution
            global_x = map_origin[0] + navigatable_point[1] * map_resolution
            global_y = map_origin[1] + navigatable_point[0] * map_resolution
            waypoints.append((global_x, global_y))
        return waypoints

    def send_next_closest_waypoint(self):
        if self.pupper_position and self.generated_waypoints and self.patrol_status:
            # find the smallest distance between a waypoint and the current position and a waypoint
            # if using RL we could get rid of this and have some more sophisticated way of finding the next waypoint to go to
            nearest_waypoint = min(self.generated_waypoints, key=lambda wp: self.distance(self.pupper_position, wp))
            # remove each waypoint once it is sent
            self.generated_waypoints.remove(nearest_waypoint)
            # send the waypoint
            self.send_waypoint_to_pupper(nearest_waypoint)

    def send_waypoint_to_pupper(self, waypoint):
        # if patrol is active then send the waypoint
        if self.patrol_status:
            # create the goal message
            goal_message = FollowWaypoints.Goal()
            # set up pose to send based on calculated waypoint
            waypoint_pose = self.format_waypoint_to_pose(waypoint)
            # set the message pose to this value
            goal_message.poses = [waypoint_pose]
            # send the waypoint to the pupper through ROS
            self.waypoint_client.wait_for_server()
            self.waypoint_client.send_goal_async(goal_message)

    def format_waypoint_to_pose(self, waypoint):
        # create pose
        waypoint_pose = PoseStamped()
        waypoint_pose.header.frame_id = "map" # for navigation goal, frame id should be map
        waypoint_pose.header.stamp = self.get_clock().now().to_msg() # attach current timestamp
        # set pose x and y to calculated values
        waypoint_pose.pose.position.x = waypoint[0]
        waypoint_pose.pose.position.y = waypoint[1]
        # set orientation to 1 (we may want to calculate or randomise this in future)
        waypoint_pose.pose.orientation.w = 1.0
        return waypoint_pose

    def distance(self, first_position, second_position):
        # distance between two points = (p1_x-p2_x)^2 + (p1_y-p2_y)^2 
        return math.sqrt((first_position[0] - second_position[0]) ** 2 + (first_position[1] - second_position[1]) ** 2)

    # open windows to monitor relevant ROS topic info
    def open_windows_to_monitor_ROS_topics(self):
        ROS_topics = ["/follow_waypoints/feedback", "/waypoint_navigator/waypoints", "/plan", "/local_plan", "/cmd_vel"]
        for topic in ROS_topics:
            # open a terminal window for each topic
            self.open_terminal_and_monitor_topic(topic)

    def open_terminal_and_monitor_topic(self, topic):
        # shell command to open terminal (gnome-termial)
        # echos the topic, and content of topic
        shell_command = f"""
        gnome-terminal -- bash -c '
            echo "Monitoring {topic}...";
            ros2 topic echo {topic} || echo "Failed to access {topic}.";
            read -p "Exit the process (press enter)...";
            exec bash'
        """
        subprocess.Popen(shell_command, shell=True)

if __name__ == '__main__':
    # initialise ROS 2
    rclpy.init()
    # setup parser to take SLAM generated map yaml path
    argument_parser = argparse.ArgumentParser(description='add the path to a map so the Pupper can navigate through it.')
    argument_parser.add_argument('map_yaml_path', type=str, help='Path to map.yaml')
    # give arguments to waypoint sender node
    arguments = argument_parser.parse_args()
    waypoint_sender = Waypoint_Sender_Node(arguments.map_yaml_path)
    # have the node active to listen to any callbacks
    rclpy.spin(waypoint_sender)
    # destroy and shutdown node
    waypoint_sender.destroy_node()
    rclpy.shutdown()
