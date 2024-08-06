
# Installation Guide for OAK-D Drivers on Mini Pupper

## Relevant Links
- [DepthAI-ROS GitHub Repository](https://github.com/luxonis/depthai-ros?tab=readme-ov-file)
- [DepthAI-ROS Documentation](https://docs-beta.luxonis.com/software/ros/depthai-ros/intro/)

**Note:** You don't need to check the above links unless the instructions below fail, in which case there might be some issues with the installation.

## Installation Steps

1. **Install the DepthAI-ROS package:**
   - Open a terminal on your Mini Pupper and execute the following command:
     ```
     sudo apt install ros-humble-depthai-ros
     ```
   - This command should also install the example applications.

2. **Setup and modify the example script:**
   - Navigate to the example directory:
     ```
     sudo nano /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart.py
     ```
   - Then copy the code from the `monitor_and_restart.py` in this repository into there.

   - This script will run the spatial example and includes functionality to restart it if it fails. This added restart functionality was written to address common issues where the feed might drop due to poor internet and hardware connections.

3. **Fix the bug in the original example:**
   - Open the YOLOv4 publisher script:
     ```
     sudo nano /opt/ros/humble/share/depthai_examples/launch/yolov4_publisher.launch.py
     ```
   - Replace the existing code with the code from `modified_yolov4_publisher.launch.py` available in this repository. This corrects a bug that was not fixed in the humble version of the code.

may have to do this also: <img width="1902" alt="image" src="https://github.com/BarrettBytes/QuadropedRobotHonoursProject/assets/131525111/435f3d9b-c713-4600-a963-78190c7ec8fa">
or : Create the Udev Rule:

   ```
sudo nano /etc/udev/rules.d/80-depthai-usb.rules
   ```
 enter and save the below in there
   ```
SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", MODE="0666"
   ```
then
   ```
sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
Unplug and Replug the Camera:
This step ensures that the new rules are applied.

## Additional Information

After following these steps, the spatial example should run more reliably on your Mini Pupper, with improved stability and automatic restart capability in case of errors.

## V3 RL update

Note we have updated to mobilenetSSD so now u have to do this
 **Setup and modify the example script:**
   - Navigate to the example directory:
     ```
     sudo nano /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py
     ```
   - Then copy the code from the `monitor_and_restart_mobile.py` in this repository into there.

**Add PASCAL NAMES:**
   - Navigate to the example directory:
     ```
     sudo nano /opt/ros/humble/share/depthai_examples/launch/PASCAL.names
     ```
   - Then copy the code from the `PASCAL.names` in this repository into there.


