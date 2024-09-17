# replace the file and run with ros2 launch ~/ros2_ws/src/mini_pupper_ros/mini_pupper_navigation/launch/navigation.launch.py namespace:=robotx map:=$HOME/map.yaml

#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    this_package = FindPackageShare('mini_pupper_navigation')

    # Paths to map and parameter files
    default_map_path = PathJoinSubstitution([this_package, 'maps', 'map.yaml'])
    nav2_param_file_path = PathJoinSubstitution([this_package, 'param', 'mini_pupper.yaml'])
    rviz_config_file_path = PathJoinSubstitution([this_package, 'rviz', 'navigation.rviz'])

    use_sim_time = LaunchConfiguration('use_sim_time', default='False')
    map = LaunchConfiguration('map', default=default_map_path)

    # List of robot namespaces
    robot_namespaces = ['robotx', 'roboty']

    # Launch argument declarations
    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation (Gazebo) clock if true'
    )

    map_launch_arg = DeclareLaunchArgument(
        'map',
        default_value=default_map_path,
        description='Full path to map file to load'
    )

    # Function to create a group of nodes for each robot
    def create_robot_nav_group(namespace):
        return GroupAction([
            # Launch the container for composable nodes
            Node(
                package='rclcpp_components',
                executable='component_container',
                name=f'{namespace}_nav2_container',
                namespace=namespace,
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            ),

            # Load the costmap components into the container
            LoadComposableNodes(
                target_container=f'{namespace}_nav2_container',
                composable_node_descriptions=[
                    # Local Costmap
                    ComposableNode(
                        package='nav2_costmap_2d',
                        plugin='nav2_costmap_2d::CostmapNode',
                        name='local_costmap',
                        namespace=namespace,
                        parameters=[nav2_param_file_path],
                        remappings=[
                            ("/tf", "/tf"),
                            ("/tf_static", "/tf_static"),
                            ("/scan", f"/{namespace}/scan"),
                            ("/odom", f"/{namespace}/odom"),
                            ("/local_costmap/costmap", f"/{namespace}/local_costmap/costmap"),
                            ("/local_costmap/costmap_updates", f"/{namespace}/local_costmap/costmap_updates"),
                        ]
                    ),
                    # Global Costmap
                    ComposableNode(
                        package='nav2_costmap_2d',
                        plugin='nav2_costmap_2d::CostmapNode',
                        name='global_costmap',
                        namespace=namespace,
                        parameters=[nav2_param_file_path],
                        remappings=[
                            ("/tf", "/tf"),
                            ("/tf_static", "/tf_static"),
                            ("/scan", f"/{namespace}/scan"),
                            ("/odom", f"/{namespace}/odom"),
                            ("/global_costmap/costmap", f"/{namespace}/global_costmap/costmap"),
                            ("/global_costmap/costmap_updates", f"/{namespace}/global_costmap/costmap_updates"),
                        ]
                    )
                ]
            ),

            # Launch AMCL for localization
            Node(
                package='nav2_amcl',
                executable='amcl',
                name='amcl',
                namespace=namespace,
                output='screen',
                parameters=[nav2_param_file_path],
                remappings=[
                    ("/tf", "/tf"),
                    ("/tf_static", "/tf_static"),
                    ("/scan", f"/{namespace}/scan"),
                    ("/odom", f"/{namespace}/odom"),
                ]
            ),

            # Launch Planner Server
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                namespace=namespace,
                output='screen',
                parameters=[nav2_param_file_path]
            ),

            # Launch Controller Server
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                namespace=namespace,
                output='screen',
                parameters=[nav2_param_file_path]
            ),

            # Launch BT Navigator
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                namespace=namespace,
                output='screen',
                parameters=[nav2_param_file_path]
            ),

            # Launch Lifecycle Manager for navigation
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                namespace=namespace,
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart': True,
                    'node_names': [
                        f'{namespace}/amcl',
                        f'{namespace}/local_costmap',
                        f'{namespace}/global_costmap',
                        f'{namespace}/planner_server',
                        f'{namespace}/controller_server',
                        f'{namespace}/bt_navigator',
                    ]
                }]
            ),

            # RViz for visualization
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                namespace=namespace,
                arguments=['-d', rviz_config_file_path],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ])

    # Generate the launch description
    return LaunchDescription([
        use_sim_time_launch_arg,
        map_launch_arg,
        *[create_robot_nav_group(namespace) for namespace in robot_namespaces]
    ])
