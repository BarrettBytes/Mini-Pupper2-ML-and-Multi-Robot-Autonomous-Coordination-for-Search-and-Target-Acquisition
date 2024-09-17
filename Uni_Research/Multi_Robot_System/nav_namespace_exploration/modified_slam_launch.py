#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# (License Text)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution
)
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Find the package share directory
    slam_package = FindPackageShare('mini_pupper_slam')

    # Define paths to configuration files
    cartographer_config_dir = PathJoinSubstitution([slam_package, 'config'])
    cartographer_config_basename = TextSubstitution(text='slam.lua')
    rviz_config_file_path = PathJoinSubstitution([slam_package, 'rviz', 'slam.rviz'])

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_launch_arg = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='False',
        description='Use simulation (Gazebo) clock if true'
    )

    return LaunchDescription([
        use_sim_time_launch_arg,
        # Group all SLAM-related nodes under the /robotx namespace
        GroupAction([
            PushRosNamespace('robotx'),
            # Cartographer Node
            Node(
                package='cartographer_ros',
                executable='cartographer_node',
                name='cartographer_node',
                output='screen',
                parameters=[
                    {'use_sim_time': use_sim_time}
                ],
                arguments=[
                    '-configuration_directory', cartographer_config_dir,
                    '-configuration_basename', cartographer_config_basename,
                    '--ros-args',
                    '--log-level', 'debug',  # Optional: for detailed logs
                    '-r', 'imu:=imu/data',
                    '-r', 'scan:=scan'
                ],
                remappings=[
                    ('imu', 'imu/data'),    # Redirect 'imu' to 'imu/data'
                    ('scan', 'scan'),       # Ensure 'scan' maps to '/robotx/scan'
                ]
            ),
            # Cartographer Occupancy Grid Node
            Node(
                package='cartographer_ros',
                executable='cartographer_occupancy_grid_node',
                name='cartographer_occupancy_grid_node',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'resolution': 0.05},
                    {'publish_period_sec': 1.0}
                ],
                remappings=[
                    ('map', '/map'),  # Publish to /robotx/map instead of global /map
                ]
            ),
        ]),
        # RViz Node (optional, can be run separately or integrated with Nav2 launch)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file_path],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
    ])
