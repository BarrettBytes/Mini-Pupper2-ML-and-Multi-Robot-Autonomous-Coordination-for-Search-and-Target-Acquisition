# Multi-Robot Inference with Namespace Support

After setting up the robots, you can download and run the `Multi_Robot_inference_robotx.py` script. Follow these steps to ensure everything runs smoothly:

1. **Booting Up the Robots**  
   Each robot can be booted up using the `run_robot.sh` script within its specific namespace. Assuming you have already downloaded `run_robot.sh` into the home directory of each robot, follow the instructions below to boot them up:

   For example, if you have three robots named `robotx`, `roboty`, and `robotz`, execute the following commands on each respective robot:

   - On the robot with the namespace `robotx`, run:
     ```bash
     ~/run_robot.sh --robotx
     ```

   - On the robot with the namespace `roboty`, run:
     ```bash
     ~/run_robot.sh --roboty
     ```

   - On the robot with the namespace `robotz`, run:
     ```bash
     ~/run_robot.sh --robotz
     ```

2. **Running the Inference Script**  
   In the computer that shares the same ROS network as the robots, you need to run the `Multi_Robot_inference_robotx.py` script for each robot. This must be done in separate terminals. Ensure you are in the folder where you downloaded `Multi_Robot_inference_robotx.py` before running these commands:

   For example, for the robots `robotx`, `roboty`, and `robotz`, run the following commands in three different terminals:

   - For `robotx`, run:
     ```bash
     python3 Multi_Robot_inference_robotx.py --namespace robotx
     ```

   - For `roboty`, run:
     ```bash
     python3 Multi_Robot_inference_robotx.py --namespace roboty
     ```

   - For `robotz`, run:
     ```bash
     python3 Multi_Robot_inference_robotx.py --namespace robotz
     ```

if that worked your done, sit back and enjoy :)
---



### Running the RL Inference
To run the RL inference with your desired namespace, use the following command:

```bash
python3 Multi_Robot_inference_robotx.py --namespace robotx
```

*(Replace `robotx` with your robot's namespace, or leave it as is if you don't have a specific namespace)*

### Important Steps for Robots with Namespaces
If your robot **has a namespace**, you must follow these steps before running the inference. If your robot **does not** have a namespace, you can skip these steps.

1. **Edit the `monitor_and_restart_mobile.py` script:**
   Open the script for editing with the following command:
   ```bash
   sudo nano /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py
   ```
   Ensure that you input the code from the camera_robotx.py file into here

2. **Edit the `mobile_publisher.launch.py` file:**
   Open the file for editing:
   ```bash
   sudo nano /opt/ros/humble/share/depthai_examples/launch/mobile_publisher.launch.py
   ```
    Ensure that you input the code from the namespace_mobile_publisher.py file into here

### Running the Monitor Script with Different Namespaces
To run the `monitor_and_restart_mobile.py` script using a different namespace (e.g., `roboty`), use the following command:

```bash
python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py --namespace roboty
```

Make sure to replace `roboty` with the appropriate namespace for your setup.

---
## Run Robot Script (run_robot.sh)

For efficiency when working with multiple robots, the `run_robot.sh` script allows you to run both the camera and bringup processes together, avoiding the need to launch them separately each time. (You must follow all the steps in the Successful_Namespace_Modification folder first if you are using robotx)

### Instructions

1. After creating the script with `sudo nano` in the desired folder, follow these steps:

   ```bash
   # First time setup (make the script executable):
   chmod +x run_robot.sh
   ```

2. To run as **robotx**:

   ```bash
   ./run_robot.sh --robotx
   ```

3. To run with the **default setup**:

   ```bash
   ./run_robot.sh
   ```

This script simplifies working with your robot by running all necessary processes in parallel!
