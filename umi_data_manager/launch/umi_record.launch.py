#!/usr/bin/env python3
"""
UMI Data Manager - Recording Launch File

Auto-increments episode number within a session.

Usage:
    ros2 launch umi_data_manager umi_record.launch.py session:=demo_01
    # → /workspace/data/raw/demo_01/episode_001/
    # Run again:
    # → /workspace/data/raw/demo_01/episode_002/
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def launch_setup(context):
    session = LaunchConfiguration('session').perform(context)
    data_dir = LaunchConfiguration('data_dir').perform(context)

    session_dir = os.path.join(data_dir, session)
    os.makedirs(session_dir, exist_ok=True)

    # Find next episode number
    existing = [d for d in os.listdir(session_dir)
                if os.path.isdir(os.path.join(session_dir, d)) and d.startswith('episode_')]
    next_num = len(existing) + 1
    episode_name = f'episode_{next_num:03d}'
    output_path = os.path.join(session_dir, episode_name)

    log_info = LogInfo(msg=[
        '\n',
        '======================================\n',
        f'UMI Recording: {session} / {episode_name}\n',
        f'Output: {output_path}\n',
        '======================================\n',
        'Press Ctrl+C to stop recording\n',
    ])

    # Use compressed image topics to avoid image_transport multi-type issues
    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', output_path,
            '/camera/camera/color/image_rect_raw/compressed',
            '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
            '/camera/camera/color/camera_info',
            '/umi/gripper_position_controller/commands',
            '/umi/joint_states',
        ],
        output='screen',
    )

    return [log_info, bag_record]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('session', description='Session name (e.g. demo_01)'),
        DeclareLaunchArgument('data_dir', default_value='/workspace/data/raw'),
        OpaqueFunction(function=launch_setup),
    ])
