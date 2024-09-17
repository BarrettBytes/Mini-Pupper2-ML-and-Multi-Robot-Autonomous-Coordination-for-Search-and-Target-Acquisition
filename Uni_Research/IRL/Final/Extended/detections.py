import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class DetectionPrinter(Node):
    def __init__(self):
        super().__init__('detection_printer')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/mobilenet_detections',  # Subscribe to the correct topic
            self.detection_callback,
            10
        )

    def detection_callback(self, msg):
        self.get_logger().info('Received detections:')
        for detection in msg.detections:
            self.get_logger().info(f'Detection: {detection}')
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                self.get_logger().info(f'  Class ID: {class_id}, Score: {score}')

def main(args=None):
    rclpy.init(args=args)
    detection_printer = DetectionPrinter()
    rclpy.spin(detection_printer)
    detection_printer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
