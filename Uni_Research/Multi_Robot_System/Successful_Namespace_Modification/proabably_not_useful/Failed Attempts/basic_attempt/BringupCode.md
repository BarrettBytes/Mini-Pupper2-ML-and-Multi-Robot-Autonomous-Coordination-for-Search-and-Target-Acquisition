The minipupper bringup code and champ bringup code has been put bellow because modifying this can be useful for namespaces


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
          [FindPackageShare('champ_bringup'), 'launch', 'bringup.launch.py']
      )
      hardware_interface_launch_path = PathJoinSubstitution(
          [FindPackageShare('mini_pupper_bringup'), 'launch', 'hardware_interface.launch.py']
      )
  
      sensors_config, ports_config = get_config()
  
      # Convert bool to str because cannot pass bool directly to launch_arguments.
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
              "rviz": "False",  # set always false to launch RViz2 with costom .rviz file
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
          PythonLaunchDescriptionSource(hardware_interface_launch_path),
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
The champ bringup code which is references is also below as this may also need to be modified for namespaces
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
  
      declare_orientation_from_imu = DeclareLaunchArgument(
          "orientation_from_imu", default_value="false", description="Take orientation from IMU data"
      )
  
      declare_rviz = DeclareLaunchArgument(
          "rviz", default_value="false", description="Launch rviz"
      )
  
      declare_rviz_ref_frame = DeclareLaunchArgument(
          "rviz_ref_frame", default_value="odom", description="Rviz ref frame"
      )
  
      declare_robot_name = DeclareLaunchArgument(
          "robot_name", default_value="/", description="Robot name"
      )
  
      declare_base_link_frame = DeclareLaunchArgument(
          "base_link_frame", default_value="base_link", description="Base link frame"
      )
  
      declare_lite = DeclareLaunchArgument(
          "lite", default_value="false", description="Lite"
      )
  
      declare_gazebo = DeclareLaunchArgument(
          "gazebo", default_value="false", description="If in gazebo"
      )
  
      declare_joint_controller_topic = DeclareLaunchArgument(
          "joint_controller_topic",
          default_value="joint_group_effort_controller/joint_trajectory",
          description="Joint controller topic",
      )
  
      declare_hardware_connected = DeclareLaunchArgument(
          "joint_hardware_connected",
          default_value="false",
          description="Whether hardware is connected",
      )
  
      declare_publish_joint_control = DeclareLaunchArgument(
          "publish_joint_control",
          default_value="true",
          description="Publish joint control",
      )
  
      declare_publish_joint_states = DeclareLaunchArgument(
          "publish_joint_states",
          default_value="true",
          description="Publish joint states",
      )
  
      declare_publish_foot_contacts = DeclareLaunchArgument(
          "publish_foot_contacts",
          default_value="true",
          description="Publish foot contacts",
      )
  
      declare_publish_odom_tf = DeclareLaunchArgument(
          "publish_odom_tf",
          default_value="true",
          description="Publish odom tf from cmd_vel estimation",
      )
  
      declare_close_loop_odom = DeclareLaunchArgument(
          "close_loop_odom", default_value="false", description=""
      )
  
      description_ld = IncludeLaunchDescription(
          PythonLaunchDescriptionSource(
              os.path.join(
                  get_package_share_directory("champ_description"),
                  "launch",
                  "description.launch.py",
              )
          ),
          launch_arguments={
              "use_sim_time": LaunchConfiguration("use_sim_time"),
              "description_path": LaunchConfiguration("description_path"),
          }.items(),
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
          remappings=[("/cmd_vel/smooth", "/cmd_vel")],
      )
  
      state_estimator_node = Node(
          package="champ_base",
          executable="state_estimation_node",
          output="screen",
          parameters=[
              {"use_sim_time": LaunchConfiguration("use_sim_time")},
              {"orientation_from_imu": LaunchConfiguration("orientation_from_imu")},
              {"urdf": Command(['xacro ', LaunchConfiguration('description_path')])},
              LaunchConfiguration('joints_map_path'),
              LaunchConfiguration('links_map_path'),
              LaunchConfiguration('gait_config_path'),
          ],
      )
  
      base_to_footprint_ekf = Node(
          package="robot_localization",
          executable="ekf_node",
          name="base_to_footprint_ekf",
          output="screen",
          parameters=[
              {"base_link_frame": LaunchConfiguration("base_link_frame")},
              {"use_sim_time": LaunchConfiguration("use_sim_time")},
              os.path.join(
                  get_package_share_directory("champ_base"),
                  "config",
                  "ekf",
                  "base_to_footprint.yaml",
              ),
          ],
          remappings=[("odometry/filtered", "odom/local")],
      )
  
      footprint_to_odom_ekf = Node(
          package="robot_localization",
          executable="ekf_node",
          name="footprint_to_odom_ekf",
          output="screen",
          parameters=[
              {"base_link_frame": LaunchConfiguration("base_link_frame")},
              {"use_sim_time": LaunchConfiguration("use_sim_time")},
              os.path.join(
                  get_package_share_directory("champ_base"),
                  "config",
                  "ekf",
                  "footprint_to_odom.yaml",
              ),
          ],
          remappings=[("odometry/filtered", "odom")],
      )
  
      rviz2 = Node(
          package='rviz2',
          namespace='',
          executable='rviz2',
          name='rviz2',
          arguments=['-d', LaunchConfiguration("rviz_path")],
          condition=IfCondition(LaunchConfiguration("rviz"))
      )
  
  
      return LaunchDescription(
          [
              declare_use_sim_time,
              declare_description_path,
              declare_rviz_path,
              declare_joints_map_path,
              declare_links_map_path,
              declare_gait_config_path,
              declare_orientation_from_imu,
              declare_rviz,
              declare_rviz_ref_frame,
              declare_robot_name,
              declare_base_link_frame,
              declare_lite,
              declare_gazebo,
              declare_joint_controller_topic,
              declare_hardware_connected,
              declare_publish_joint_control,
              declare_publish_joint_states,
              declare_publish_foot_contacts,
              declare_publish_odom_tf,
              declare_close_loop_odom,
              description_ld,
              quadruped_controller_node,
              state_estimator_node,
              base_to_footprint_ekf,
              footprint_to_odom_ekf,
              rviz2
          ]
      )

```
