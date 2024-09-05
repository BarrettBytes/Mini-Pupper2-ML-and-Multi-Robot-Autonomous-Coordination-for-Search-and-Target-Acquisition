# Circlar Walking 

"""
The goal of this script is to implement the object detection and circular walking for minipupper 2 robot. 

The script aims to give a circular path for robot while the robot is doing object detection. it could maximise the searching area in grid-based environmrnt and increase the searching efficiency of finding cat and dog.

### Databases
This code uses SQLite databases to store and generate data such that it doesnt need to all be held in RAM
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray

# Giving Circular path
class CircularMotionNode(Node):
    def __init__(self):
        """
    	initialise the circular path parameter, such as velocity, the radius of the circular path
        """
        super().__init__('circular_motion_node')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 3) # set the velocity of 3ms
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.declare_parameter('linear_velocity', 0.5)
        self.declare_parameter('radius', 0.5) #set the circular radius

        self.linear_vel = self.get_parameter('linear_velocity').get_parameter_value().double_value
        self.radius = self.get_parameter('radius').get_parameter_value().double_value

    def timer_callback(self):
        """
        set the function for circular path and callback to the minipupper
        """
        
        msg = Twist()
        angular_vel = self.linear_vel / self.radius
        msg.linear.x = self.linear_vel
        msg.angular.z = angular_vel
        self.publisher.publish(msg)

# Given Object Detection, detecting dog and cat
class DetectionProcessor(Node):
    def __init__(self):
        """
        initialise the circular path parameter, such as velocity, the radius of the circular path
        """

        super().__init__('detection_processor')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/mobilenet_detections',
            self.detection_callback,
            10
        )
        self.mobilenet_labels = self.load_mobilenet_labels()

    def load_mobilenet_labels(self):   
        """
        load the labels dataset from mobilenet
        """

        return [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
        ]

    def detection_callback(self, msg):
        """
        set the function for object detection and call back to minipupper
        """   
    
        for detection in msg.detections:
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                if score > 0.5 and class_id < len(self.mobilenet_labels):  # Adjusted threshold to 0.5
                    detected_object = self.mobilenet_labels[class_id]
                    print(f"Detected: {detected_object} (Score: {score:.2f})")

def main(args=None):

    """
    Main function to run the cicular walking and object detection
    """
    rclpy.init(args=args)
    node = CircularMotionNode()
    rclpy.spin(node)
    detection_processor = DetectionProcessor()
    
    print("Detection Processor started. Listening for detections...")
    try:
        rclpy.spin(node)
        rclpy.spin(detection_processor)
    except KeyboardInterrupt: #stop the minipupper by interrupt the keyboard
        pass
    finally:
        detection_processor.destroy_node() #shutdown this scipt when it detect the dog
        rclpy.shutdown()

if __name__ == '__main__':
    main()
