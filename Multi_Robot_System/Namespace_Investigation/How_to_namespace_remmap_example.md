Here’s the complete README with your requested additions, including the command to source the workspace setup and launch the test.
This modifies the two functions in https://github.com/BarrettBytes/QuadropedRobotHonoursProject/blob/main/Multi_Robot_System/Namespace_Investigation/BringupCode.md 
which are responsible for setting up all of the nodes, you may notice a didn't work folder here, this was an attempt to combine champ_bringup
and mini_pupper_bringup, this is essentially impossible because of how the builds are made and you links.yaml will never be read properly.

The correct approach is to make a file next to the mini_pupper_bringup which is an exact copy of the mini puppe bringup
then modify it so it no longer points to the original champ_bringup but a new one. This new one should be an exact copy of the old one in the exact same folder
it should have a new name and you can then modify the mappings. In this example cmd_vel is remmaped to robotx/cmd_vel

---

# README: Steps to Modify and Launch ROS2 Setup for Mini Pupper

This README will guide you through the steps to modify the ROS2 launch files and use the `ros2 launch` command properly for your Mini Pupper setup.

## Prerequisites
Ensure that you have already installed ROS2 and the `champ_bringup` package in the `ros2_ws` workspace. Also, ensure that the file paths and dependencies for `mini_pupper_ros` and `champ_bringup` are set up correctly.

## Step 1: Modify `bringup_launch_remapp.py`

1. Open the `bringup_launch_remapp.py` file for editing using `sudo nano`:

    ```bash
    sudo nano ~/ros2_ws/install/champ_bringup/share/champ_bringup/launch/bringup_launch_remapp.py
    ```

2. Replace the content of the file with the following code:

    ```python
    import os
    import xacro
    import launch_ros
    import xml.etree.ElementTree as ET
    from ament_index_python.packages import get_package_share_directory
    from launch_ros.actions import Node
    from launch import LaunchDescription
    from launch.actions import (
        DeclareLaunchArgument,
        ExecuteProcess,
        IncludeLaunchDescription,
        GroupAction,
        RegisterEventHandler
    )
    from launch.event_handlers.on_process_exit import OnProcessExit
    from launch.event_handlers.on_execution_complete import OnExecutionComplete
    from launch.conditions import IfCondition
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch.substitutions import Command, LaunchConfiguration

    def generate_launch_description():

        config_pkg_share = launch_ros.substitutions.FindPackageShare(
            package="champ_config"
        ).find("champ_config")
        descr_pkg_share = launch_ros.substitutions.FindPackageShare(
            package="champ_description"
        ).find("champ_description")

        joints_config = os.path.join(config_pkg_share, "config/joints/joints.yaml")
        gait_config = os.path.join(config_pkg_share, "config/gait/gait.yaml")
        links_config = os.path.join(config_pkg_share, "config/links/links.yaml")

        default_rviz_path = os.path.join(descr_pkg_share, "rviz/urdf_viewer.rviz")
        default_model_path = os.path.join(descr_pkg_share, "urdf/champ.urdf.xacro")

        declare_use_sim_time = DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true",
        )

        declare_description_path = DeclareLaunchArgument(
            name="description_path",
            default_value=default_model_path,
            description="Absolute path to robot urdf file",
        )

        declare_rviz_path = DeclareLaunchArgument(
            name="rviz_path",
            default_value=default_rviz_path,
            description="Absolute path to rviz file",
        )

        declare_joints_map_path = DeclareLaunchArgument(
            name="joints_map_path",
            default_value='',
            description="Absolute path to joints map file",
        )

        declare_links_map_path = DeclareLaunchArgument(
            name="links_map_path",
            default_value='',
            description="Absolute path to links map file",
        )

        declare_gait_config_path = DeclareLaunchArgument(
            name="gait_config_path",
            default_value='',
            description="Absolute path to gait config file",
        )

        quadruped_controller_node = Node(
            package="champ_base",
            executable="quadruped_controller_node",
            output="screen",
            parameters=[
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
                {"gazebo": LaunchConfiguration("gazebo")},
                {"publish_joint_states": LaunchConfiguration("publish_joint_states")},
                {"publish_joint_control": LaunchConfiguration("publish_joint_control")},
                {"publish_foot_contacts": LaunchConfiguration("publish_foot_contacts")},
                {"joint_controller_topic": LaunchConfiguration("joint_controller_topic")},
                {"urdf": Command(['xacro ', LaunchConfiguration('description_path')])},
                LaunchConfiguration('joints_map_path'),
                LaunchConfiguration('links_map_path'),
                LaunchConfiguration('gait_config_path'),
            ],
        )

        return LaunchDescription(
            [
                declare_use_sim_time,
                declare_description_path,
                declare_rviz_path,
                declare_joints_map_path,
                declare_links_map_path,
                declare_gait_config_path,
                quadruped_controller_node
            ]
        )
    ```

3. Save the file and exit (`Ctrl + O`, then `Ctrl + X`).

## Step 2: Modify `bringup_basic_remap.launch.py`

1. Open the `bringup_basic_remap.launch.py` file using `sudo nano`:

    ```bash
    sudo nano ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/launch/bringup_basic_remap.launch.py
    ```

2. Replace the content of the file with the following code:

    ```python
    #!/usr/bin/env python3

    # SPDX-License-Identifier: Apache-2.0
    #
    # Copyright (c) 2022-2023 MangDang
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    import os
    import yaml
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
    from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.substitutions import FindPackageShare
    from launch.conditions import IfCondition
    from ament_index_python.packages import get_package_share_directory

    ROBOT_MODEL = os.getenv('ROBOT_MODEL', default="mini_pupper_2")

    def get_config():
        bringup_package = get_package_share_directory('mini_pupper_bringup')
        config_file_name = ROBOT_MODEL + '.yaml'
        config_file_path = os.path.join(bringup_package, 'config', config_file_name)

        with open(config_file_path, 'r') as f:
            configuration = yaml.safe_load(f)

        sensors_config = configuration.get('sensors', {})
        sensors_config.setdefault('lidar', False)
        sensors_config.setdefault('imu', False)
        sensors_config.setdefault('camera', False)

        ports_config = configuration.get('ports', {})

        return sensors_config, ports_config

    def generate_launch_description():

        description_package = FindPackageShare('mini_pupper_description')

        description_path = PathJoinSubstitution(
            [description_package, 'urdf', ROBOT_MODEL, 'mini_pupper_description.urdf.xacro']
        )

        joints_config_path = PathJoinSubstitution(
            [description_package, 'config', 'champ', ROBOT_MODEL, 'joints.yaml']
        )
        links_config_path = PathJoinSubstitution(
            [description_package, 'config', 'champ', ROBOT_MODEL, 'links.yaml']
        )
        gait_config_path = PathJoinSubstitution(
            [description_package, 'config', 'champ', ROBOT_MODEL, 'gait.yaml']
        )

        champ_bringup_launch_path = PathJoinSubstitution(
            ['/home/ubuntu/ros2_ws/install/champ_bringup/share/champ_bringup/launch', 'bringup_launch_remapp.py']
        )

        hardware_interface_launch_path = PathJoinSubstitution(
            [FindPackageShare('mini_pupper_bringup'), 'launch', 'hardware_interface.launch.py']
        )

        sensors_config, ports_config = get_config()

        has_lidar = str(sensors_config['lidar'])
        has_imu = str(sensors_config['imu'])
        has_camera = str(sensors_config['camera'])
        lidar_port = ports_config['lidar']

        use_sim_time = LaunchConfiguration('use_sim_time')
        use_sim_time_launch_arg = DeclareLaunchArgument(
            name='use_sim_time',
            default_value='False',
            description='Use simulation (Gazebo) clock if true'
        )

        hardware_connected = LaunchConfiguration("hardware_connected")
        hardware_connected_launch_arg = DeclareLaunchArgument(
            name='hardware_connected',
            default_value='True',
            description='Set to true if connected to a physical robot'
        )

        champ_bringup_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(champ_bringup_launch_path),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "robot_name": ROBOT_MODEL,
                "gazebo": use_sim_time,
                "rviz": "False",
                "joint_hardware_connected": hardware_connected,
                "orientation_from_imu": has_imu,
                "publish_foot_contacts": "True",
                "close_loop_odom": "True",
                "joint_controller_topic": "joint_group_effort_controller/joint_trajectory",
                "joints_map_path": joints_config_path,
                "links_map_path": links_config_path,
                "gait_config_path": gait_config_path,
                "description_path": description_path
            }.items()
        )

        hardware_interface_launch = IncludeLaunchDescription(
            PythonLaunchDescription

Source(hardware_interface_launch_path),
            condition=IfCondition(hardware_connected),
            launch_arguments={
                "has_lidar": has_lidar,
                "has_imu": has_imu,
                "has_camera": has_camera,
                "lidar_port": lidar_port
            }.items()
        )

        return LaunchDescription([
            use_sim_time_launch_arg,
            hardware_connected_launch_arg,
            champ_bringup_launch,
            hardware_interface_launch
        ])
    ```

3. Save the file and exit (`Ctrl + O`, then `Ctrl + X`).

## Step 3: Test the Configuration

Once the two files have been modified, you need to source your ROS2 workspace before running the launch command.

1. Source the workspace:

    ```bash
    . ~/ros2_ws/install/setup.bash
    ```

2. Run the ROS2 launch command to start the Mini Pupper setup:

    ```bash
    ros2 launch ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/launch/bringup_basic_remap.launch.py
    ```

This command should start the necessary nodes and bring up the Mini Pupper with the specified configuration.

---

By following these steps, you will successfully modify the launch files and be able to launch and test your Mini Pupper setup with the provided configurations.

This has remmapped /cmd_vel to /robotx/cmd_vel

test it with
   ```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/robotx/cmd_vel
  ```


For future changes where you want to remmap something note
    ```bash
    ros2 topic list
    ```
will show you all the topics currently in existance in ros

   ```bash
    ros2 topic info -v /topic_name
   ```
will give you the publishing and subscribing info
this is essential because you need to know which nodes are publishing and subscribing to something to know how to remapp it

