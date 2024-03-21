Please be sure that you have completed the tasks in the following before doing this  [NetworkSetup.md](NetworkSetup.md), [ROS_install_on_pupper_device.md](ROS_install_on_pupper_device.md), [Time_Sync.md](Time_Sync.md)

## SLAM (Mapping)

### Launch Mini Pupper
```bash
# SSH to the Mini Pupper and run the following command
source ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_bringup bringup.launch.py
```

### Launch SLAM on the PC
```bash
# On the PC, run the following command
source ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_navigation slam.launch.py
```

you should be able to launch RVIZ with, if not please install it
```bash
rviz2
```

### Control Mini Pupper using the Keyboard
Utilize the keyboard to remotely control the Mini Pupper and facilitate the mapping process.
```bash
# On the PC, run the following command in a new terminal
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Save the Generated Map
The map will be saved in the `$HOME` directory.
```bash
# On the PC, run the following commands in a new terminal
ros2 service call /finish_trajectory cartographer_ros_msgs/srv/FinishTrajectory "{trajectory_id: 0}"
ros2 service call /write_state cartographer_ros_msgs/srv/WriteState "{filename: '${HOME}/cartographer_map.pbstream'}"
ros2 run nav2_map_server map_saver_cli -f ${HOME}/cartographer_map
```

---

## Navigation

### Launch Mini Pupper
```bash
# SSH to the Mini Pupper and run the following command
source ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_bringup bringup.launch.py
```

### Update the Map Files
Replace the `cartographer_map.pbstream` file in the maps folder with the newly generated `cartographer_map.pbstream`.
```bash
# On the PC, run the following commands
cp -f ~/cartographer_map.pgm ~/ros2_ws/src/mini_pupper_ros/mini_pupper_navigation/maps/cartographer_map.pgm
cp -f ~/cartographer_map.pbstream ~/ros2_ws/src/mini_pupper_ros/mini_pupper_navigation/maps/cartographer_map.pbstream
cp -f ~/cartographer_map.yaml ~/ros2_ws/src/mini_pupper_ros/mini_pupper_navigation/maps/cartographer_map.yaml
```

#### Launch Localization and Mapping

```bash
# Terminal 3
. ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_navigation bringup.launch.py
```

## Possible other things to try


### Launch Localization by iteslf
```bash
# On the PC, run the following command in a new terminal
source ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_navigation localization.launch.py
```

### Launch Navigation by itself
```bash
# On the PC, run the following command in a new terminal
source ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_navigation navigation.launch.py
```

### Launch RViz (Much better to just run to just run the Launch Localization and Mapping command)
```bash
# On the PC, run the following command in a new terminal
source ~/ros2_ws/install/setup.bash # Use setup.zsh if using zsh instead of bash
ros2 launch mini_pupper_bringup bringup.launch.py joint_hardware_connected:=true rviz:=true robot_name:=mini_pupper_2
```

### RVIS environment
The following video gives some tips for setting up RVIS, remember to add the map with the \map topic
https://www.youtube.com/watch?v=0CsSok3QgZk
