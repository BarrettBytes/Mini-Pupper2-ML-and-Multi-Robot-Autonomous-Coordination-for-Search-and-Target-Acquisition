# starts the yolov4_publisher.launch.py monitors it for disconnection
# which is common due to wifi and hardware issues
# then restarts it

# TODO this code is very hacky as the subprocess was behaving very micheviously
# it was a hard problem to get working
# as this isn't core it has been left in it's current fit for purpose state
# it would be good to impove this to make it less hacky
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
            '/color/yolov4_detections',
            self.listener_callback,
            1)
        self.message_received = False

    def listener_callback(self, msg):
        print("detected")
        self.message_received = True

def launch_process():
    return subprocess.Popen(['ros2', 'launch', 'depthai_examples', 'yolov4_publisher.launch.py'])

def monitor_topic(monitor, stop_event):
    attempts = 0
    monitor.message_received = False
    while attempts < 20 and not stop_event.is_set():
        rclpy.spin_once(monitor, timeout_sec=1)  # Process callbacks for up to 10 seconds
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
