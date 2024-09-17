# Mini Pupper ROS2 Setup Guide

This guide provides step-by-step instructions for setting up the Mini Pupper robot with ROS2 Humble on Ubuntu.

## Prerequisites

- Ubuntu operating system
- Internet connection

## Installation Steps

1. Install ROS
  
    ```bash
    cd ~
    git clone https://github.com/mangdangroboticsclub/mini_pupper_ros.git -b ros2-dev mini_pupper_ros
    ```

    ```bash
    cd mini_pupper_ros
    ./install.sh
    ```

2. Update the package list and install the required dependencies:

   ```bash
   cd ~
   sudo apt-get update
   sudo apt -y install python3-pip python3-venv python3-virtualenv
   ```

3. Clone the `ros2_setup_scripts_ubuntu` repository (this step may be unnecessary as step 1 may have done this):

   ```bash
   git clone https://github.com/Tiryoh/ros2_setup_scripts_ubuntu.git
   ```

4. Run the ROS2 Humble setup script:

   ```bash
   ~/ros2_setup_scripts_ubuntu/ros2-humble-ros-base-main.sh
   ```

5. Source the ROS2 setup file:

   ```bash
   source /opt/ros/humble/setup.bash
   ```

6. Create a workspace directory and navigate to the `src` folder:

   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

7. Clone the `mini_pupper_ros` repository:

   ```bash
   git clone --depth 1 https://github.com/mangdangroboticsclub/mini_pupper_ros.git -b ros2-dev mini_pupper_ros
   ```

8. Import the required repositories:

   ```bash
   vcs import < mini_pupper_ros/.minipupper.repos --recursive
   ```

9. Ignore certain packages to avoid compiling Gazebo and Cartographer on Raspberry Pi (this is for install on the robot, you may need to skip this for pc install):

   ```bash
   touch champ/champ/champ_gazebo/AMENT_IGNORE
   touch champ/champ/champ_navigation/AMENT_IGNORE
   touch mini_pupper_ros/mini_pupper_gazebo/AMENT_IGNORE
   touch mini_pupper_ros/mini_pupper_navigation/AMENT_IGNORE
   ```

10. Navigate to the workspace directory:

   ```bash
   cd ~/ros2_ws
   ```

11. Install the required dependencies:

    ```bash
    rosdep install --from-paths src --ignore-src -r -y --skip-keys=joint_state_publisher_gui --skip-keys=rviz2 --skip-keys=gazebo_plugins --skip-keys=velodyne_gazebo_plugins
    ```

12. Install the `teleop-twist-keyboard` package:

    ```bash
    sudo apt install ros-humble-teleop-twist-keyboard
    ```

13. Build the workspace:

    ```bash
    MAKEFLAGS=-j1 colcon build --executor sequential --symlink-install
    ```

## Usage

1. In Terminal 1 (ssh), source the setup file and launch the Mini Pupper bringup:

   ```bash
   . ~/ros2_ws/install/setup.bash # setup.zsh if you use zsh instead of bash
   ros2 launch mini_pupper_bringup bringup.launch.py
   ```

2. In Terminal 2 (ssh), run the `teleop_twist_keyboard` node:

   ```bash
   ros2 run teleop_twist_keyboard teleop_twist_keyboard
   ```

3. Control the Mini Pupper robot using the keyboard.

## Troubleshooting

- If you encounter any issues during the setup process, try deleting the `~/ros2_ws` directory and repeating the installation steps:

  ```bash
  rm -rf ~/ros2_ws
  ```
- If step 1 and 2 had issues you may need to delete ros2_setup_scripts_ubuntu and mini_pupper_ros in a similar manner and start over

- Please note that the network card on the Mini Pupper robot may be unreliable. If the internet connection is weak, the setup process might fail. In such cases, delete the workspace and try again.

For more information and support, please refer to the official Mini Pupper documentation and community resources.
