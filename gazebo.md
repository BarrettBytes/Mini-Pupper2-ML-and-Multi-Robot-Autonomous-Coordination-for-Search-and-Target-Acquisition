Here's the modified instructions in Markdown format:

## 2.1.3 Test SLAM (Mapping) in Gazebo

Note: This step is only for PC

* Bring up Gazebo

  ```bash
  # Terminal 1
  . ~/ros2_ws/install/setup.bash
  ros2 launch mini_pupper_gazebo gazebo.launch.py
  ```

* Mapping on PC

  ```bash
  # Terminal 2
  . ~/ros2_ws/install/setup.bash
  ros2 launch mini_pupper_navigation slam.launch.py use_sim_time:=true
  ```

* Keyboard control Use the keyboard to remotely control the Mini Pupper to complete the mapping.

  ```bash
  # Terminal 3
  ros2 run teleop_twist_keyboard teleop_twist_keyboard
  ```

* Save the map The map will be saved in the `gazebo_maps` folder.

  ```bash
  # Terminal 4 (on PC)

  # Create the "gazebo_maps" folder if it doesn't exist
  mkdir -p "${HOME}/gazebo_maps"

  ros2 service call /finish_trajectory cartographer_ros_msgs/srv/FinishTrajectory "{trajectory_id: 0}"

  ros2 service call /write_state cartographer_ros_msgs/srv/WriteState "{filename: '${HOME}/gazebo_maps/cartographer_map.pbstream'}"

  ros2 run nav2_map_server map_saver_cli -f ${HOME}/gazebo_maps/cartographer_map
  ```

### 2.1.4 Test Navigation in Gazebo

* Replace the map files Remember to replace the `cartographer_map.pbstream` in the maps folder with your new `cartographer_map.pbstream` first.

  ```bash
  # On the PC, run the following commands

  cp -f ~/gazebo_maps/cartographer_map.pgm ~/ros2_ws/src/mini_pupper_ros/mini_pupper_navigation/maps/cartographer_map.pgm

  cp -f ~/gazebo_maps/cartographer_map.pbstream ~/ros2_ws/src/mini_pupper_ros/mini_pupper_navigation/maps/cartographer_map.pbstream

  cp -f ~/gazebo_maps/cartographer_map.yaml ~/ros2_ws/src/mini_pupper_ros/mini_pupper_navigation/maps/cartographer_map.yaml
  ```

* Bring up Gazebo

  ```bash
  # Terminal 1
  . ~/ros2_ws/install/setup.bash
  ros2 launch mini_pupper_gazebo gazebo.launch.py
  ```

* Localization & Navigation

  ```bash
  # Terminal 3
  . ~/ros2_ws/install/setup.bash
  ros2 launch mini_pupper_navigation bringup.launch.py use_sim_time:=true
  ```

* Localization only (This step is not necessary if you run `bringup.launch.py`.)

  ```bash
  # Terminal 4
  . ~/ros2_ws/install/setup.bash
  ros2 launch mini_pupper_navigation localization.launch.py use_sim_time:=true
  ```

* Navigation only (This step is not necessary if you run `bringup.launch.py`.)

  ```bash
  # Terminal 4
  . ~/ros2_ws/install/setup.bash
  ros2 launch mini_pupper_navigation navigation.launch.py use_sim_time:=true
  ```
  
# Terminal 1
  ```bash
. ~/ros2_ws/install/setup.bash # setup.zsh if you use zsh instead of bash
ros2 launch mini_pupper_bringup bringup.launch.py joint_hardware_connected:=false rviz:=true robot_name:=mini_pupper_2
  ```
