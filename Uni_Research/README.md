# Quadroped Robot Honours Project

## NetworkSetup.md

[NetworkSetup.md](NetworkSetup.md)

## Time_Sync.md

[Time_Sync.md](Time_Sync.md)

## ROS_install_on_pupper_device.md

[ROS_install_on_pupper_device.md](ROS_install_on_pupper_device.md)

## Run SLAM_and_Navigation.md

[SLAM_and_Navigation.md](SLAM_and_Navigation.md)


** Random things u can do **
* stop battery sound:
  
  sudo systemctl stop battery_monitor 

* screen not working for some reason but robot on:
  
  if robot on network find ip with
  
  nmap -p 22 --open -sV 192.168.1.0/24
  
  look for a line in the output like: Nmap scan report for <some_ip>
  
  then type

  ssh ubuntu@<some_ip>


  all the repos mangdang uses are:

  repositories:
  main/mini_pupper:
    type: git
    url: https://github.com/mangdangroboticsclub/mini_pupper_ros.git
    version: ros2
  champ/champ:
    type: git
    url: https://github.com/mangdangroboticsclub/champ.git
    version: ros2
  champ/champ_teleop:
    type: git
    url: https://github.com/chvmp/champ_teleop.git
    version: ros2
  ldlidar_stl_ros:
    type: git
    url: https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git
    version: master
