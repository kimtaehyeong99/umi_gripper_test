#!/usr/bin/env python3
"""
UMI Data Manager - SLAM Processing Launch File

Runs ORB-SLAM3 + bag replay in the same launch (same Zenoh session)
to collect camera poses, then triggers data conversion.

Usage:
    ros2 launch umi_data_manager umi_slam.launch.py \
        bag_path:=/workspace/data/raw/demo_01 \
        output_path:=/workspace/data/processed/demo_01
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'bag_path',
            description='Path to recorded ROS2 bag'),
        DeclareLaunchArgument(
            'output_path',
            description='Output path for processed HDF5'),
        DeclareLaunchArgument(
            'bag_rate', default_value='1.0',
            description='Bag playback rate'),
        DeclareLaunchArgument(
            'settings_name', default_value='RealSense_D405',
            description='ORB-SLAM3 settings name'),
    ]

    bag_path = LaunchConfiguration('bag_path')
    output_path = LaunchConfiguration('output_path')

    log_info = LogInfo(msg=[
        '\n',
        '======================================\n',
        'UMI SLAM Processing\n',
        '  Bag: ', bag_path, '\n',
        '  Output: ', output_path, '\n',
        '======================================\n',
    ])

    # ORB-SLAM3 node
    slam_node = Node(
        package='ros2_orb_slam3',
        executable='rgbd_node_cpp',
        name='rgbd_node_cpp',
        parameters=[{
            'settings_name': LaunchConfiguration('settings_name'),
            'rgb_topic': '/camera/camera/color/image_rect_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'enable_viewer': False,
        }],
        output='screen',
    )

    # Bag replay (start after SLAM is ready, 3s delay)
    bag_play = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    bag_path,
                    '--rate', LaunchConfiguration('bag_rate'),
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription(declared_arguments + [
        log_info,
        slam_node,
        bag_play,
    ])
