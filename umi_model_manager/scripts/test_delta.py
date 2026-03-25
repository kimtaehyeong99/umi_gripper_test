#!/usr/bin/env python3
"""
Test: send a simple delta pose to the motion controller.
Run INSIDE Docker (same terminal context as controller).

Usage:
  python3 test_delta.py            # +1cm in X
  python3 test_delta.py --axis z   # +1cm in Z
  python3 test_delta.py --axis x --dist 0.05  # +5cm in X
"""

import argparse
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class DeltaTestNode(Node):
    def __init__(self, axis, dist, topic, frame_id):
        super().__init__('delta_test')
        self.pub = self.create_publisher(PoseStamped, topic, 10)
        self.axis = axis
        self.dist = dist
        self.frame_id = frame_id
        self.sent = 0
        self.timer = self.create_timer(0.5, self._send)

    def _send(self):
        if self.sent >= 3:
            self.get_logger().info('Done sending. Ctrl+C to exit.')
            self.timer.cancel()
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # Identity rotation (no rotation change)
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        # Position delta
        if self.axis == 'x':
            msg.pose.position.x = self.dist
        elif self.axis == 'y':
            msg.pose.position.y = self.dist
        elif self.axis == 'z':
            msg.pose.position.z = self.dist

        self.pub.publish(msg)
        self.sent += 1
        self.get_logger().info(
            f'[{self.sent}/3] Sent delta {self.axis}={self.dist:.3f}m '
            f'on {self.pub.topic_name} frame={self.frame_id}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axis', default='x', choices=['x', 'y', 'z'])
    parser.add_argument('--dist', type=float, default=0.01)
    parser.add_argument('--topic', default='/odom/camera_pose')
    parser.add_argument('--frame', default='odom')
    args = parser.parse_args()

    rclpy.init()
    node = DeltaTestNode(args.axis, args.dist, args.topic, args.frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
