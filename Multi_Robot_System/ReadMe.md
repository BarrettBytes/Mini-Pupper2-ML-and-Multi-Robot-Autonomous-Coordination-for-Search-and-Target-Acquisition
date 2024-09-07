Multi Robot System is to complement with object detection, navigation(still developing)/given path pattern with three robots, so they can communicate each othere and sending signal if one of the robots find the cat. 

For example, if one robot find the cat, it will the signal to other two robot to stop and sending the final corridnate of cat back to PC.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Cicrular Walking
if you want to run this code. Please run the following commands. 

Step 1: Open Terminal (Terminal 1), you need to ssh to minipuipper robot. 
~~~
ssh ubuntu@[your_minipupper_id]
~~~

Step 2: bringup the robot to your PC.
~~~
. ~/ros2_ws/install/setup.bash # setup.zsh if you use zsh instead of bash
ros2 launch mini_pupper_bringup bringup.launch.py
~~~

Step 3: Open a new Terminal(Termianl 2), repeated Step 1, and run this command to activate the ROS2 Depthai.
~~~
cd depthai_examples/launch/share/depthai_examples/launch
python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py
~~~

Step 4: Run Circular Walking
~~~
cd [The location you saved this code]
Python3 Circular Walking.py
~~~

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# SLAM 
if you want to run this code. Please run the following commands. 

Step 1: Open Terminal (Terminal 1), you need to ssh to minipuipper robot. 
~~~
ssh ubuntu@[your_minipupper_id]
~~~

Step 2: bringup the robot to your PC.
~~~
. ~/ros2_ws/install/setup.bash # setup.zsh if you use zsh instead of bash
ros2 launch mini_pupper_bringup bringup.launch.py
~~~

Step 3: Open a new terminal from Remote PC with Ctrl + Alt + T and launch the SLAM node. 
~~~
. ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_slam slam.launch.py
~~~

# Navigation 
Step 1: Open a terminal with Ctrl+Alt+T on remote PC. Run teleoperation node using the following command. (keyboard - teleoperation)
~~~
. ~/ros2_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
~~~

Step 2: Open a terminal with Ctrl+Alt+T on remote PC. Use the following command to launch the map_saver_cli node in the nav2_map_server package to create map files. (save the map)

The map file is saved in the directory where the map_saver_cli node is launched at.

~~~
. ~/ros2_ws/install/setup.bash
ros2 run nav2_map_server map_saver_cli -f ~/map
~~~

After running the above command, two files will be generated, namely map.pgm and map.yaml.

The following pictures shows the .pgm file to be saved.

![map2](https://github.com/user-attachments/assets/cefa9a20-2874-4e29-9cb4-6464db7a19bc)

Step 3: Open a new terminal from Remote PC with Ctrl + Alt + T and launch the Navigation node.
~~~
. ~/ros2_ws/install/setup.bash
ros2 launch mini_pupper_navigation navigation.launch.py map:=$HOME/map.yaml
~~~

The map used in navigation is two-dimensional Occupancy Grid Map (OGM). The white area is collision free area while black area is occupied and inaccessible area, and gray area represents the unknown area.
