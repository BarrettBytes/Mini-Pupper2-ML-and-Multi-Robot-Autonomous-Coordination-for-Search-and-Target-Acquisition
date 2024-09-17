import os
import time
import subprocess
from threading import Thread, Event
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class CameraMonitor(Node):
    def __init__(self, namespace='roboty'):
        super().__init__('camera_monitor', namespace=namespace)
        self.namespace = namespace  # Store namespace for topic subscription
        self.subscription = self.create_subscription(
            Detection2DArray,
            f'/{self.namespace}/color/mobilenet_detections', 
            self.listener_callback,
            1)
        self.message_received = False
        self.PASCAL_labels = self.load_pascal_labels('PASCAL.names')

    def load_pascal_labels(self, filepath):
        # PASCAL VOC labels
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

    def listener_callback(self, msg):
        self.message_received = True
        for detection in msg.detections:
            for result in detection.results:
                class_id = int(result.hypothesis.class_id)
                score = result.hypothesis.score
                if score > 0.5 and class_id < len(self.PASCAL_labels):
                    detected_object = self.PASCAL_labels[class_id]
                    print(f'{detected_object} detected with confidence {score:.2f}')

def launch_process(namespace='roboty'):
    """
    Launch the mobile_publisher.launch.py with the specified namespace.
    """
    launch_args = ['ros2', 'launch', 'depthai_examples', 'mobile_publisher.launch.py']
    
    if namespace:
        launch_args += [f'namespace:={namespace}']
    
    return subprocess.Popen(launch_args)

def monitor_topic(monitor, stop_event):
    attempts = 0
    monitor.message_received = False
    while attempts < 20 and not stop_event.is_set():
        rclpy.spin_once(monitor, timeout_sec=1)  # Process callbacks for up to 1 second
        if monitor.message_received:
            print("received")
            monitor.message_received = False  # Reset the flag after receiving a message
            attempts = 0  # Reset attempts on successful message
        else:
            attempts += 1
            print(f"No message received in the last second. Attempt {attempts}/20")
        time.sleep(1)

    if attempts >= 20 and not stop_event.is_set():
        print("No message received during the last 20 seconds. Initiating restart...")
        stop_event.set()

def main(args=None):
    rclpy.init(args=args)
    
    namespace = 'roboty'  # Set your desired namespace here

    monitor = CameraMonitor(namespace=namespace)
    stop_event = Event()
   
    while not stop_event.is_set():
        process = launch_process(namespace=namespace)
        monitor_thread = Thread(target=monitor_topic, args=(monitor, stop_event))
        monitor_thread.start()
        monitor_thread.join()

        if stop_event.is_set():
            print("Terminating the mobile_publisher.launch.py process...")
            process.terminate()
            process.wait()
            stop_event.clear()
            print("Restarting in 5 seconds...")
            time.sleep(5)  # Delay before restarting the process

    monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
