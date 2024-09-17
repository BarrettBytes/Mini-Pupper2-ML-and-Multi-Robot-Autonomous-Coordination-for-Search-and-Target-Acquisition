import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    
    # dcmdbot1_config = os.path.join(get_package_share_directory('localization_server'), 'config', 'dcmdbot4_amcl_config.yaml')
    namespace = 'robotx'
    # namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')

    this_package = FindPackageShare('mini_pupper_navigation')

    # Define file paths for map, parameters, and RViz
    # default_map_path = PathJoinSubstitution([this_package, 'maps', 'map.yaml'])
    params_file = PathJoinSubstitution([this_package, 'param', 'mini_pupper.yaml'])
    # remappings = [('/tf', 'tf'),
    #               ('/tf_static', 'tf_static')]
    log_level="info"
    return LaunchDescription([
 
        # Node(
        #     namespace='dcmdbot4',
        #     package='nav2_amcl',
        #     executable='amcl',
        #     name='amcl',
        #     output='screen',
        #     parameters=[dcmdbot1_config]
        # ),
        Node(
                package='nav2_amcl',
                namespace=namespace,
                executable='amcl',
                name='amcl',
                output='screen',
                respawn=False,
                respawn_delay=2.0,
                parameters=[params_file],
                arguments=['--ros-args', '--log-level', log_level]),
                # remappings=remappings),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{'use_sim_time': False},
                        {'autostart': True},
                        {'bond_timeout':0.0},
                        {'node_names':  [namespace+ '/amcl']}]
        )
    ])
