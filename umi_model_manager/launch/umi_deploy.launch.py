#!/usr/bin/env python3
"""
UMI Model Manager - Deploy Launch File

Launches motion controller + policy bridge for robot deployment.

Usage:
    ros2 launch umi_model_manager umi_deploy.launch.py
    ros2 launch umi_model_manager umi_deploy.launch.py zmq_host:=127.0.0.1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('dry_run', default_value='false'),
        DeclareLaunchArgument('zmq_host', default_value='127.0.0.1'),
        DeclareLaunchArgument('zmq_port', default_value='8766'),
        DeclareLaunchArgument('launch_controller', default_value='true'),
        DeclareLaunchArgument('record_bag', default_value='false'),
        DeclareLaunchArgument('bag_path', default_value='/tmp/umi_deploy_debug'),
    ]

    config_file = PathJoinSubstitution([
        FindPackageShare('umi_model_manager'), 'config', 'bridge_config.yaml'
    ])

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('motion_controller_ros'), 'controller.launch.py'
            ])
        ]),
        launch_arguments={'controller_type': 'right_arm_relative'}.items(),
        condition=IfCondition(LaunchConfiguration('launch_controller')),
    )

    bridge_node = Node(
        package='umi_model_manager',
        executable='bridge_node',
        name='umi_model_manager',
        parameters=[
            config_file,
            {
                'dry_run': LaunchConfiguration('dry_run'),
                'zmq_host': LaunchConfiguration('zmq_host'),
                'zmq_port': LaunchConfiguration('zmq_port'),
            }
        ],
        output='screen',
    )

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', LaunchConfiguration('bag_path'),
            '/r_gripper_pose', '/joint_states', '/odom/camera_pose',
            '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory',
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record_bag')),
    )

    return LaunchDescription(declared_arguments + [
        controller_launch,
        bridge_node,
        bag_record,
    ])
