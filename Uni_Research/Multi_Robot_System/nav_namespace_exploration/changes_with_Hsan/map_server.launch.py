import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    map_file = os.path.join(get_package_share_directory('mini_pupper_navigation'), 'maps', 'map.yaml')
#     remappings = [('/tf', 'tf'),
#                   ('/tf_static', 'tf_static')]
    log_level="info"
    return LaunchDescription([
        # Node(
        #     package='nav2_map_server',
        #     executable='map_server',
        #     name='map_server',
        #     output='screen',
        #     parameters=[{'use_sim_time': False}, 
        #                 {'yaml_filename':map_file} 
        #                ]),
        Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                output='screen',
                respawn=False,
                respawn_delay=2.0,
                parameters=[{'use_sim_time': False}, 
                        {'yaml_filename':map_file}],
                arguments=['--ros-args', '--log-level', log_level]),
                # remappings=remappings),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_mapper',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{'use_sim_time': False},
                        {'autostart': True},
                        {'node_names': ['map_server']}])            
        ])