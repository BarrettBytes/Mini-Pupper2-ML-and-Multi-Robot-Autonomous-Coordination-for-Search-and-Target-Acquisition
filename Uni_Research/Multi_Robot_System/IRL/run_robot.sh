#!/bin/bash

# first time u run will need to 
# sudo chmod +x run_robot.sh

# to run as robotx
# ./run_robot.sh --robotx

# to run as normal
# ./run_robot.sh 

# Check if --robotx argument is provided
if [[ "$1" == "--robotx" ]]; then
    echo "Running for --robotx"

    # Source ROS2 environment and run the first command with remap in the background
    . ~/ros2_ws/install/setup.bash
    ros2 launch ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/launch/bringup_basic_with_remap.launch.py &

    # Run the second python script in the background
    python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py &

# Check if --robotx argument is provided
elif [[ "$1" == "--roboty" ]]; then
    echo "Running for --roboty"

    # Source ROS2 environment and run the first command with remap in the background
    # make sure u edited bringup with remap to be roboty
    . ~/ros2_ws/install/setup.bash
    ros2 launch ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/launch/bringup_with_remap.launch.py &

    # Run the second python script in the background
    python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py &

else
    echo "Running default setup"

    # Source ROS2 environment and run the default bringup command in the background
    . ~/ros2_ws/install/setup.bash
    ros2 launch mini_pupper_bringup bringup.launch.py &

    # Run the second python script in the background
    python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py &
fi

# Wait for both processes to finish (if needed)
wait
