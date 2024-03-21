# Controlling Mini Pupper with an Xbox Controller in Gazebo

This README provides step-by-step instructions on how to set up and control the Mini Pupper robot using an Xbox controller in the Gazebo simulation environment.

## Prerequisites

- Mini Pupper ROS package installed and set up
- ROS 2 (Humble) installed
- Gazebo simulation environment installed

## Step 1: Install Required Packages

Open a terminal and run the following command to install the necessary packages:

```
sudo apt install ros-humble-joy ros-humble-teleop-twist-joy
```

## Step 2: Connect Xbox Controller

Connect your Xbox controller to your computer via USB or wireless adapter. Verify that the controller is recognized by running the following command:

```
ls /dev/input/js*
```

If the controller is recognized, you should see a device file like `/dev/input/js0`.

## Step 3: Create Configuration File

1. Navigate to the `mini_pupper_bringup` directory in your Mini Pupper ROS package:

```
cd ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup
```

2. Create a new file named `xbox_controller_config.yaml` and open it in a text editor:

```
nano xbox_controller_config.yaml
```

3. Add the following content to the file:

```yaml
teleop_twist_joy:
  ros__parameters:
    axis_linear: 1 # Left thumb stick vertical axis
    scale_linear: 0.5
    axis_angular: 0 # Left thumb stick horizontal axis
    scale_angular: 1.0
    enable_button: 0 # A button
```

4. Save the file and exit the text editor.

## Step 4: Launch Nodes

1. Open a new terminal and source your ROS workspace:

```
source ~/ros2_ws/install/setup.bash
```

2. Launch the `joy` node to read the controller inputs:

```
ros2 run joy joy_node
```

3. In another terminal, launch the `teleop_twist_joy` node with the configuration file:

```
ros2 run teleop_twist_joy teleop_node --ros-args --params-file ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/xbox_controller_config.yaml
```

4. Verify that the `teleop_twist_joy` node is publishing `Twist` messages:

```
ros2 topic echo /cmd_vel
```

Move the left thumb stick on the Xbox controller and check if the linear and angular velocities change accordingly.

## Step 5: Launch Gazebo Simulation

In a new terminal, launch the Mini Pupper robot in Gazebo simulation:

```
ros2 launch mini_pupper_gazebo gazebo.launch.py
```

## Step 6: Control Mini Pupper

Once the Gazebo simulation is running, you should be able to control the Mini Pupper robot using the Xbox controller. Move the left thumb stick to control the robot's linear and angular velocities.

## Troubleshooting

If you encounter any issues, check the following:

- Ensure that the Xbox controller is properly connected and recognized by the system.
- Verify that the `joy` node is running and publishing messages on the `/joy` topic.
- Make sure the `teleop_twist_joy` node is running with the correct configuration file and publishing messages on the `/cmd_vel` topic.
- Confirm that the Gazebo simulation is running and the Mini Pupper robot is properly spawned.

If the `ros2 topic echo /cmd_vel` command is not showing your controller inputs, try the following steps:

1. Stop the `teleop_twist_joy` node:
   - In the terminal where the `teleop_twist_joy` node is running, press `Ctrl+C` to stop the node.

2. Stop the `joy` node:
   - In the terminal where the `joy` node is running, press `Ctrl+C` to stop the node.

3. Relaunch the `joy` node:
   - Open a new terminal and source your ROS 2 workspace:
     ```
     source ~/ros2_ws/install/setup.bash
     ```
   - Launch the `joy` node:
     ```
     ros2 run joy joy_node
     ```

4. Relaunch the `teleop_twist_joy` node:
   - Open a new terminal and source your ROS 2 workspace:
     ```
     source ~/ros2_ws/install/setup.bash
     ```
   - Launch the `teleop_twist_joy` node with the configuration file:
     ```
     ros2 run teleop_twist_joy teleop_node --ros-args --params-file ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/xbox_controller_config.yaml
     ```

5. Verify the `/cmd_vel` topic:
   - In another terminal, run the following command to check if the `teleop_twist_joy` node is publishing `Twist` messages:
     ```
     ros2 topic echo /cmd_vel
     ```
   - Move the left thumb stick on the Xbox controller and see if the linear and angular velocities change accordingly.

If you have any further questions or need additional assistance, please refer to the Mini Pupper ROS documentation or seek help from the ROS community.
