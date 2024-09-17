import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray

class CameraMonitor(Node):
    def __init__(self):
        super().__init__('camera_monitor')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/yolov4_detections',
            self.listener_callback,
            10)
        self.bottle_detected_pub = self.create_publisher(Bool, 'bottle_detected', 10)
        self.bottle_detected = False

        # Load COCO labels
        with open("yolo/coco.names", "r") as file:
            self.coco_labels = [line.strip() for line in file.readlines()]

    def listener_callback(self, msg):
        if self.bottle_detected == False:
            detected_bottle = False
            for detection in msg.detections:
                for result in detection.results:
                    class_id = int(result.hypothesis.class_id)
                    confidence = int(result.hypothesis.score)
                    print(f"class_id: {class_id}")  
                    print(str(len(self.coco_labels)))
                    label = self.coco_labels[class_id]
                    print(f"Detected {label} with confidence {confidence:.2f}")  # Print what was detected
                    if label == "bottle" and confidence > 0.5:
                        print(f"Detected bottle with confidence {confidence:.2f}")
                        detected_bottle = True
                        break
                if detected_bottle:
                    self.bottle_detected = True 
                    break
        
        if detected_bottle:
            self.bottle_detected_pub.publish(Bool(data=True))
            print("Bottle successfully detected! Signal sent to stop the robot.")


def main(args=None):
    rclpy.init(args=args)
    monitor = CameraMonitor()
    rclpy.spin(monitor)  # This will keep your node alive and responsive to callbacks
    monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
