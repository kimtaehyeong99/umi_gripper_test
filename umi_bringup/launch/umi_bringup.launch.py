"""
UMI Data Pipeline - Data Collection Launch File

Launches:
1. Dynamixel gripper (hardware_dual)
2. RealSense D405 camera with aligned depth

Usage:
    ros2 launch umi_data_pipeline data_collection.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    align_depth_arg = DeclareLaunchArgument(
        'align_depth',
        default_value='true',
        description='Enable depth alignment to color frame'
    )

    depth_profile_arg = DeclareLaunchArgument(
        'depth_profile',
        default_value='640x480x30',
        description='Depth stream profile (WxHxFPS)'
    )

    color_profile_arg = DeclareLaunchArgument(
        'color_profile',
        default_value='640x480x30',
        description='Color stream profile (WxHxFPS)'
    )

    # Dynamixel gripper launch
    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('dynamixel_hardware_interface_example_4'),
                'launch',
                'hardware_dual.launch.py'
            ])
        ]),
    )

    # RealSense camera launch
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'align_depth.enable': LaunchConfiguration('align_depth'),
            'depth_module.depth_profile': LaunchConfiguration('depth_profile'),
            'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
            'rgb_camera.color_format': 'RGB8',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
        }.items()
    )

    # Log info
    log_info = LogInfo(msg=[
        '\n',
        '======================================\n',
        'UMI Data Collection Ready\n',
        '======================================\n',
        'Gripper: Dynamixel dual system\n',
        'Camera: RealSense D405\n',
        'Resolution: ', LaunchConfiguration('color_profile'), '\n',
        '======================================\n',
        '\n',
        'To start recording, run in another terminal:\n',
        '  cd /root/ros2_ws/src/umi_gripper_test/umi_data_pipeline\n',
        '  ./scripts/record_raw_data.sh data/raw/demo_XX\n',
        '\n',
    ])

    return LaunchDescription([
        align_depth_arg,
        depth_profile_arg,
        color_profile_arg,
        gripper_launch,
        realsense_launch,
        log_info,
    ])
