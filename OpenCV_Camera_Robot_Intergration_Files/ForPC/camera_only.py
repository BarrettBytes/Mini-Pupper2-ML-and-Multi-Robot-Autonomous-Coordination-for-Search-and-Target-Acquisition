import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class DetectionProcessor(Node):
    def __init__(self):
        super().__init__('detection_processor')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/mobilenet_detections',
            self.detection_callback,
            10
        )
        self.mobilenet_labels = self.load_mobilenet_labels()

    def load_mobilenet_labels(self):
        return [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
        ]

    def detection_callback(self, msg):
        for detection in msg.detections:
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                if score > 0.5 and class_id < len(self.mobilenet_labels):  # Adjusted threshold to 0.5
                    detected_object = self.mobilenet_labels[class_id]
                    print(f"Detected: {detected_object} (Score: {score:.2f})")

def main(args=None):
    rclpy.init(args=args)
    detection_processor = DetectionProcessor()
    print("Detection Processor started. Listening for detections...")
    try:
        rclpy.spin(detection_processor)
    except KeyboardInterrupt:
        pass
    finally:
        detection_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()