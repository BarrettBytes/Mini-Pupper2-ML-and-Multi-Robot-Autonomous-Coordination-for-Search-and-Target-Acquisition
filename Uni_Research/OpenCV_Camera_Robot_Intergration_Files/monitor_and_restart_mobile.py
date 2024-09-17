import os
import time
import subprocess
from threading import Thread, Event
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class CameraMonitor(Node):
    def __init__(self):
        super().__init__('camera_monitor')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/color/mobilenet_detections', 
            self.listener_callback,
            1)
        self.message_received = False
        self.PASCAL_labells = self.load_coco_labels('PASCAL.names')

    def load_coco_labels(self, filepath):
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
                if score > 0.5:
                    detected_object = self.PASCAL_labells[class_id]
                    print(f'{detected_object} detected with confidence {score:.2f}')

def launch_process():
    return subprocess.Popen(['ros2', 'launch', 'depthai_examples', 'mobile_publisher.launch.py'])

def monitor_topic(monitor, stop_event):
    attempts = 0
    monitor.message_received = False
    while attempts < 20 and not stop_event.is_set():
        rclpy.spin_once(monitor, timeout_sec=1)  # Process callbacks for up to 1 second
        if monitor.message_received:
           print("received")
        time.sleep(1)
        attempts += 1

    if not monitor.message_received and not stop_event.is_set():
        print("No message received during the last 20 seconds. Initiating restart...")
        stop_event.set()

def main(args=None):
    rclpy.init(args=args)
    monitor = CameraMonitor()
    stop_event = Event()
   
    while not stop_event.is_set():
        process = launch_process()
        monitor_thread = Thread(target=monitor_topic, args=(monitor, stop_event))
        monitor_thread.start()
        monitor_thread.join()

        if stop_event.is_set():
            process.terminate()
            process.wait()
            stop_event.clear()
            print("Restarting in 5 seconds...")
            time.sleep(5)  # Delay before restarting the process

    monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
