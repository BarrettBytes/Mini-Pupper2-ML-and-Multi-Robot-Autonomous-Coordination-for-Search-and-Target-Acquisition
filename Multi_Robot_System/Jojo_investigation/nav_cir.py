import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from vision_msgs.msg import Detection2DArray

class CircularMotionNode(Node):
    def __init__(self):
        """
        Initialize the circular path parameters, such as velocity and radius.
        """
        super().__init__('circular_motion_node')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 3)  # set the velocity of 3ms
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.declare_parameter('linear_velocity', 0.5)
        self.declare_parameter('radius', 0.5)  # set the circular radius

        self.linear_vel = self.get_parameter('linear_velocity').get_parameter_value().double_value
        self.radius = self.get_parameter('radius').get_parameter_value().double_value

    def timer_callback(self):
        """
        Set the function for circular path and callback to the MiniPupper.
        """
        msg = Twist()
        angular_vel = self.linear_vel / self.radius
        msg.linear.x = self.linear_vel
        msg.angular.z = angular_vel
        self.publisher.publish(msg)

class DetectionProcessor(Node):
    def __init__(self):
        """
        Initialize the object detection node.
        """
        super().__init__('detection_processor')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/mobilenet_detections',
            self.detection_callback,
            10
        )
        self.mobilenet_labels = self.load_mobilenet_labels()
        self.coord_publisher = self.create_publisher(PointStamped, '/final_coordinates', 10)  # Publisher for final coordinates

    def load_mobilenet_labels(self):
        """
        Load the labels dataset from MobileNet.
        """
        return [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
        ]

    def detection_callback(self, msg):
        """
        Process detected objects and log the results.
        """
        for detection in msg.detections:
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                if score > 0.5 and class_id < len(self.mobilenet_labels):  # Adjusted threshold to 0.5
                    detected_object = self.mobilenet_labels[class_id]
                    print(f"Detected: {detected_object} (Score: {score:.2f})")
                    if detected_object in ["cat", "dog"]:
                        # Send the final coordinates
                        self.send_final_coordinates(result.hypothesis.position)

    def send_final_coordinates(self, position):
        """
        Publish the final coordinates to the /final_coordinates topic.
        """
        coord_msg = PointStamped()
        coord_msg.header.frame_id = "map"
        coord_msg.header.stamp = self.get_clock().now().to_msg()
        coord_msg.point.x = position.x
        coord_msg.point.y = position.y
        coord_msg.point.z = 0.0  # Assuming 2D coordinates, set z to 0

        self.get_logger().info(f"Sending final coordinates: ({coord_msg.point.x}, {coord_msg.point.y})")
        self.coord_publisher.publish(coord_msg)

class NavigationGoalPublisher(Node):
    def __init__(self):
        """
        Initialize the navigation goal publisher node.
        """
        super().__init__('navigation_goal_publisher')
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        self.goal_y = self.get_parameter('goal_y').get_parameter_value().double_value

        self.timer = self.create_timer(5.0, self.publish_goal)

    def publish_goal(self):
        """
        Publish a goal position to the navigation stack.
        """
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = self.goal_x
        goal_msg.pose.position.y = self.goal_y
        goal_msg.pose.orientation.w = 1.0  # Default orientation

        self.get_logger().info(f"Publishing navigation goal: ({self.goal_x}, {self.goal_y})")
        self.goal_publisher.publish(goal_msg)

def main(args=None):
    """
    Main function to run the circular walking, object detection, and navigation.
    """
    rclpy.init(args=args)

    circular_motion_node = CircularMotionNode()
    detection_processor = DetectionProcessor()
    navigation_goal_publisher = NavigationGoalPublisher()

    try:
        rclpy.spin(circular_motion_node)
        rclpy.spin(detection_processor)
        rclpy.spin(navigation_goal_publisher)
    except KeyboardInterrupt:  # Stop the MiniPupper by interrupting the keyboard
        pass
    finally:
        # Destroy nodes and shutdown
        circular_motion_node.destroy_node()
        detection_processor.destroy_node()
        navigation_goal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
