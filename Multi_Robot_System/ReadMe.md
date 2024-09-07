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
