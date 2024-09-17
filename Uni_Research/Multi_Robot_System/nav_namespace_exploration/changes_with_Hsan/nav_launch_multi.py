# Copyright (c) 2018 Intel Corporation
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

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('nav2_bringup')

    # namespace = str(LaunchConfiguration('namespace'))
    namespace = 'robotx'
    # use_namespace = LaunchConfiguration('use_namespace')
    log_level="info"

    params_file = LaunchConfiguration('params_file')

    # autostart = LaunchConfiguration('autostart')
    # params_file = LaunchConfiguration('params_file')
    # use_composition = LaunchConfiguration('use_composition')
    # container_name = LaunchConfiguration('container_name')
    # container_name_full = (namespace, '/', container_name)
    # use_respawn = LaunchConfiguration('use_respawn')
    # log_level = LaunchConfiguration('log_level')

    lifecycle_nodes = [namespace+ '/controller_server',
                       namespace + '/smoother_server',
                       namespace + '/planner_server',
                       namespace + '/behavior_server',
                       namespace + '/bt_navigator',
                       namespace + '/waypoint_follower',
                       namespace + '/velocity_smoother']

    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='robotx',
        description='Top-level namespace')


    # declare_use_sim_time_cmd = DeclareLaunchArgument(
    #     'use_sim_time',
    #     default_value='false',
    #     description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'params', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    # declare_autostart_cmd = DeclareLaunchArgument(
    #     'autostart', default_value='true',
    #     description='Automatically startup the nav2 stack')

    # declare_use_respawn_cmd = DeclareLaunchArgument(
    #     'use_respawn', default_value='False',
    #     description='Whether to respawn if a node crashes. Applied when composition is disabled.')

    # declare_log_level_cmd = DeclareLaunchArgument(
    #     'log_level', default_value='info',
    #     description='log level')

    # nav_yaml_dcmdbot3 = os.path.join(get_package_share_directory('path_planner_server'), 'config', 'dcmdbot3_nav.yaml')

    load_nodes = GroupAction(
        # condition=IfCondition(PythonExpression(use_namespace)),
        actions=[
            Node(
                package='nav2_controller',
                namespace=namespace,
                executable='controller_server',
                output='screen',
                # respawn=False,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level]),
                # remappings=[('cmd_vel', 'cmd_vel_nav')]),
            Node(
                package='nav2_smoother',
                namespace=namespace,
                executable='smoother_server',
                name='smoother_server',
                output='screen',
                # respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_planner',
                namespace=namespace,
                executable='planner_server',
                name='planner_server',
                output='screen',
                # respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                # remappings=remappings,
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_behaviors',
                namespace=namespace,
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                # respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_bt_navigator',
                namespace=namespace,
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                # respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_waypoint_follower',
                namespace=namespace,
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                # respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_velocity_smoother',
                namespace=namespace,
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                # respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level]),
                # remappings= [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name= namespace+ '_lifecycle_manager_navigation',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                parameters=[{'use_sim_time': False},
                            {'autostart': False},
                            {'bond_timeout':0.0},
                            {'node_names': lifecycle_nodes}]),
        ]
    )

    
    # Create the launch description and populate
    ld = LaunchDescription()

    # Set environment variables
    ld.add_action(stdout_linebuf_envvar)

    # Declare the launch options
    ld.add_action(declare_namespace_cmd)
    # ld.add_action(declare_use_namespace_cmd)
    # ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    # ld.add_action(declare_autostart_cmd)
    # ld.add_action(declare_use_composition_cmd)
    # ld.add_action(declare_container_name_cmd)
    # ld.add_action(declare_use_respawn_cmd)
    # ld.add_action(declare_log_level_cmd)
    # Add the actions to launch all of the navigation nodes
    ld.add_action(load_nodes)

    return ld
