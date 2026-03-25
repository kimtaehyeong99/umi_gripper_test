#!/usr/bin/env python3
"""
Send robot arm to initial pose via JointTrajectory.

Usage:
  ros2 run umi_policy_bridge init_pose_node
  ros2 run umi_policy_bridge init_pose_node --ros-args -p duration:=3.0
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class InitPoseNode(Node):
    def __init__(self):
        super().__init__('init_pose_node')

        self.declare_parameter('duration', 3.0)
        self.declare_parameter('joint_positions', [0.75, 0.0, 0.0, -2.3, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('topic', '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory')

        duration = self.get_parameter('duration').value
        positions = self.get_parameter('joint_positions').value
        topic = self.get_parameter('topic').value

        joint_names = [
            'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4',
            'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', 'gripper_r_joint1',
        ]

        self.pub = self.create_publisher(JointTrajectory, topic, 10)

        # Build trajectory message
        msg = JointTrajectory()
        msg.joint_names = joint_names[:len(positions)]

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.velocities = [0.0] * len(positions)
        sec = int(duration)
        nanosec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]

        # Publish after short delay (wait for subscribers)
        self.msg = msg
        self.timer = self.create_timer(1.0, self._publish)
        self.sent = False

    def _publish(self):
        if not self.sent:
            self.pub.publish(self.msg)
            positions = self.msg.points[0].positions
            self.get_logger().info(
                f'Sent init pose: {[round(p, 3) for p in positions]} '
                f'(duration={self.msg.points[0].time_from_start.sec}s)')
            self.sent = True
        else:
            # Publish once more for reliability, then shutdown
            self.pub.publish(self.msg)
            self.get_logger().info('Done. Shutting down.')
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = InitPoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
