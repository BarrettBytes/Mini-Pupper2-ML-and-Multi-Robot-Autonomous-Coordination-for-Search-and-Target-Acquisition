import os
import yaml
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

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

    # Hardcoded joints_map
    joints_map = {
        "left_front": [
            "base_lf1",
            "lf1_lf2",
            "lf2_lf3",
            "lf3_lffoot"
        ],
        "right_front": [
            "base_rf1",
            "rf1_rf2",
            "rf2_rf3",
            "rf3_rffoot"
        ],
        "left_hind": [
            "base_lb1",
            "lb1_lb2",
            "lb2_lb3",
            "lb3_lbfoot"
        ],
        "right_hind": [
            "base_rb1",
            "rb1_rb2",
            "rb2_rb3",
            "rb3_rbfoot"
        ]
    }

    # Hardcoded links_map (original content from links.yaml)
    links_map = {
        'base': 'base_link',
        'left_front': ['lf1', 'lf2', 'lf3', 'lffoot'],
        'right_front': ['rf1', 'rf2', 'rf3', 'rffoot'],
        'left_hind': ['lb1', 'lb2', 'lb3', 'lbfoot'],
        'right_hind': ['rb1', 'rb2', 'rb3', 'rbfoot']
    }

    # Hardcoded gait configuration (from gait.yaml)
    gait_config = {
        "knee_orientation": ">>",
        "pantograph_leg": False,
        "odom_scaler": 1.0,
        "max_linear_velocity_x": 0.15,
        "max_linear_velocity_y": 0.15,
        "max_angular_velocity_z": 1.0,
        "com_x_translation": 0.0,
        "swing_height": 0.017,
        "stance_depth": 0.0,
        "stance_duration": 0.2,
        "nominal_height": 0.06
    }

    # Launch the hardware interface path
    hardware_interface_launch_path = PathJoinSubstitution(
        [FindPackageShare('mini_pupper_bringup'), 'launch', 'hardware_interface.launch.py']
    )

    # Get config
    sensors_config, ports_config = get_config()

    # Convert bool to str because LaunchArguments cannot accept booleans directly
    has_lidar = str(sensors_config['lidar'])
    has_imu = str(sensors_config['imu'])
    has_camera = str(sensors_config['camera'])
    lidar_port = ports_config['lidar']

    # Launch arguments
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

    declare_description_path = DeclareLaunchArgument(
        name="description_path",
        default_value=PathJoinSubstitution(
            [FindPackageShare('mini_pupper_description'), 'urdf', ROBOT_MODEL, 'mini_pupper_description.urdf.xacro']
        ),
        description="Absolute path to robot URDF file"
    )

    declare_orientation_from_imu = DeclareLaunchArgument(
        "orientation_from_imu", default_value="false", description="Take orientation from IMU data"
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
        description="Joint controller topic"
    )

    declare_hardware_connected = DeclareLaunchArgument(
        "joint_hardware_connected",
        default_value="false",
        description="Whether hardware is connected"
    )

    declare_publish_joint_control = DeclareLaunchArgument(
        "publish_joint_control",
        default_value="true",
        description="Publish joint control"
    )

    declare_publish_joint_states = DeclareLaunchArgument(
        "publish_joint_states",
        default_value="true",
        description="Publish joint states"
    )

    declare_publish_foot_contacts = DeclareLaunchArgument(
        "publish_foot_contacts",
        default_value="true",
        description="Publish foot contacts"
    )

    declare_publish_odom_tf = DeclareLaunchArgument(
        "publish_odom_tf",
        default_value="true",
        description="Publish odom tf from cmd_vel estimation"
    )

    declare_close_loop_odom = DeclareLaunchArgument(
        "close_loop_odom", default_value="false", description=""
    )

    # Nodes and other launch inclusions
    description_ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("champ_description"),
                "launch",
                "description.launch.py"
            )
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "description_path": LaunchConfiguration("description_path")
        }.items()
    )

    # Main quadruped controller node with debugging
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
            {"joints_map": joints_map},  # Hardcoded joints map
            {"links_map": links_map},  # Hardcoded links map
            {"gait_config": gait_config}  # Hardcoded gait configuration
        ],
        remappings=[("/cmd_vel/smooth", "/cmd_vel")],
        arguments=['--ros-args', '--log-level', 'DEBUG']  # Increased logging
    )

    state_estimator_node = Node(
        package="champ_base",
        executable="state_estimation_node",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"orientation_from_imu": LaunchConfiguration("orientation_from_imu")},
            {"urdf": Command(['xacro ', LaunchConfiguration('description_path')])},
            {"joints_map": joints_map},  # Hardcoded joints map
            {"links_map": links_map},  # Hardcoded links map
            {"gait_config": gait_config}  # Hardcoded gait configuration
        ],
        arguments=['--ros-args', '--log-level', 'DEBUG']  # Increased logging
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
                "base_to_footprint.yaml"
            )
        ],
        remappings=[("odometry/filtered", "odom/local")]
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
                "footprint_to_odom.yaml"
            )
        ],
        remappings=[("odometry/filtered", "odom")]
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

    return LaunchDescription(
        [
            use_sim_time_launch_arg,
            hardware_connected_launch_arg,
            declare_description_path,
            declare_orientation_from_imu,
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
            hardware_interface_launch
        ]
    )
