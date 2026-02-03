"""
UMI Data Pipeline - Data Collection Launch File

Launches:
1. RealSense D405 camera with aligned depth
2. (Optional) Gripper driver - should be launched separately if needed

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
        }.items()
    )

    # Log info
    log_info = LogInfo(msg=[
        '\n',
        '======================================\n',
        'UMI Data Collection Started\n',
        '======================================\n',
        'Camera: RealSense D405\n',
        'Resolution: ', LaunchConfiguration('color_profile'), '\n',
        '======================================\n',
        '\n',
        'To start recording, run in another terminal:\n',
        '  ros2 run umi_data_pipeline record_raw_data.sh <session_name>\n',
        '\n',
    ])

    return LaunchDescription([
        align_depth_arg,
        depth_profile_arg,
        color_profile_arg,
        log_info,
        realsense_launch,
    ])
