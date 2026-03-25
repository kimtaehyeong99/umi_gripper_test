#!/usr/bin/env python3
"""
UMI Policy Bridge Visualizer (standalone, runs on HOST outside Docker)

Shows real-time:
- Camera observation image (what the policy sees)
- EEF position XYZ history + target trajectory
- Raw action values bar chart per step

Usage (from host, with ROS2 sourced):
  python3 visualizer_standalone.py

Press 'q' to quit.
"""

import collections
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float64MultiArray


class PolicyVisualizer(Node):
    def __init__(self):
        super().__init__('policy_visualizer')

        self.latest_image = None
        self.latest_eef_pose = None
        self.latest_targets = None
        self.latest_raw_actions = None

        self.max_history = 300
        self.eef_history_x = collections.deque(maxlen=self.max_history)
        self.eef_history_y = collections.deque(maxlen=self.max_history)
        self.eef_history_z = collections.deque(maxlen=self.max_history)

        self.create_subscription(Image, '/umi_policy_bridge/debug/obs_image', self._image_cb, 5)
        self.create_subscription(PoseStamped, '/r_gripper_pose', self._eef_cb, 10)
        self.create_subscription(PoseArray, '/umi_policy_bridge/debug/target_poses', self._targets_cb, 5)
        self.create_subscription(Float64MultiArray, '/umi_policy_bridge/debug/raw_actions', self._actions_cb, 5)

        self.create_timer(0.1, self._render)
        self.get_logger().info('Visualizer started. Press q to quit.')

    def _image_cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _eef_cb(self, msg):
        p = msg.pose.position
        self.latest_eef_pose = (p.x, p.y, p.z)
        self.eef_history_x.append(p.x)
        self.eef_history_y.append(p.y)
        self.eef_history_z.append(p.z)

    def _targets_cb(self, msg):
        self.latest_targets = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]

    def _actions_cb(self, msg):
        data = np.array(msg.data)
        n_steps = len(data) // 10
        if n_steps > 0:
            self.latest_raw_actions = data.reshape(n_steps, 10)

    def _draw_plot(self, canvas, region, histories, targets_per_axis, colors, labels):
        """Draw time-series plot with target dots."""
        rx, ry, rw, rh = region
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (30, 30, 30), -1)

        all_vals = []
        for h in histories:
            if len(h) > 0:
                all_vals.extend(list(h))
        if targets_per_axis:
            for tlist in targets_per_axis:
                all_vals.extend(tlist)

        if len(all_vals) < 2:
            return

        vmin = min(all_vals) - 0.01
        vmax = max(all_vals) + 0.01
        if vmax - vmin < 0.005:
            mid = (vmax + vmin) / 2
            vmin, vmax = mid - 0.005, mid + 0.005

        def val_to_y(v):
            return int(ry + rh - (v - vmin) / (vmax - vmin) * rh)

        # Grid lines
        for frac in [0.25, 0.5, 0.75]:
            gy = ry + int(rh * frac)
            cv2.line(canvas, (rx, gy), (rx + rw, gy), (50, 50, 50), 1)
            gval = vmax - frac * (vmax - vmin)
            cv2.putText(canvas, f'{gval:.3f}', (rx + 2, gy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

        # Plot histories
        for hist, color, label in zip(histories, colors, labels):
            arr = np.array(hist)
            n = len(arr)
            if n < 2:
                continue
            for i in range(1, n):
                x1 = rx + int((i - 1) / self.max_history * rw)
                x2 = rx + int(i / self.max_history * rw)
                y1 = np.clip(val_to_y(arr[i - 1]), ry, ry + rh)
                y2 = np.clip(val_to_y(arr[i]), ry, ry + rh)
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
            cv2.putText(canvas, f'{label}={arr[-1]:.4f}', (rx + rw - 120, ry + 15 + labels.index(label) * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Draw target dots
        if targets_per_axis:
            for tvals, color in zip(targets_per_axis, colors):
                for j, tv in enumerate(tvals):
                    ty = np.clip(val_to_y(tv), ry, ry + rh)
                    tx = rx + rw + 4 + j * 5
                    cv2.circle(canvas, (min(tx, rx + rw + 50), ty), 3, color, -1)

    def _draw_action_bars(self, canvas, region, actions):
        """Draw bar chart of raw action pos values per step."""
        rx, ry, rw, rh = region
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (30, 30, 30), -1)

        n_steps = len(actions)
        pos_vals = actions[:, :3]
        max_abs = max(np.abs(pos_vals).max(), 0.001)

        group_w = rw // n_steps
        bar_w = max(2, (group_w - 6) // 3)
        mid_y = ry + rh // 2
        cv2.line(canvas, (rx, mid_y), (rx + rw, mid_y), (60, 60, 60), 1)

        colors = [(0, 0, 255), (0, 220, 0), (255, 120, 0)]
        for si in range(n_steps):
            gx = rx + si * group_w + 4
            for ai in range(3):
                val = actions[si, ai]
                bh = int(val / max_abs * (rh // 2 - 5))
                bh = max(-(rh // 2 - 5), min(rh // 2 - 5, bh))
                bx = gx + ai * (bar_w + 1)
                if bh >= 0:
                    cv2.rectangle(canvas, (bx, mid_y - bh), (bx + bar_w, mid_y), colors[ai], -1)
                else:
                    cv2.rectangle(canvas, (bx, mid_y), (bx + bar_w, mid_y - bh), colors[ai], -1)

            cv2.putText(canvas, str(si), (gx + bar_w, ry + rh - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        cv2.putText(canvas, f'scale=+/-{max_abs:.4f}', (rx + rw - 130, ry + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
        cv2.putText(canvas, f'grip={actions[0, 9]:.3f}', (rx + rw - 130, ry + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    def _render(self):
        H, W = 620, 1060
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Left: observation image
        if self.latest_image is not None:
            img = cv2.resize(self.latest_image, (400, 400))
            canvas[10:410, 10:410] = img
        else:
            cv2.putText(canvas, 'No image', (150, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        cv2.putText(canvas, 'Policy Observation', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Bottom left: EEF + target text
        if self.latest_eef_pose:
            x, y, z = self.latest_eef_pose
            cv2.putText(canvas, f'EEF:    x={x:+.4f}  y={y:+.4f}  z={z:+.4f}',
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        if self.latest_targets and len(self.latest_targets) > 0:
            t = self.latest_targets[0]
            cv2.putText(canvas, f'Tgt[0]: x={t[0]:+.4f}  y={t[1]:+.4f}  z={t[2]:+.4f}',
                        (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 180, 255), 1)
            if len(self.latest_targets) > 7:
                t7 = self.latest_targets[7]
                cv2.putText(canvas, f'Tgt[7]: x={t7[0]:+.4f}  y={t7[1]:+.4f}  z={t7[2]:+.4f}',
                            (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 140, 255), 1)

        if self.latest_raw_actions is not None:
            a0 = self.latest_raw_actions[0, :3]
            cv2.putText(canvas, f'Act[0]: x={a0[0]:+.5f}  y={a0[1]:+.5f}  z={a0[2]:+.5f}',
                        (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

        # Right top: EEF position plot
        tgt_axes = None
        if self.latest_targets:
            tgt_axes = [
                [t[0] for t in self.latest_targets],
                [t[1] for t in self.latest_targets],
                [t[2] for t in self.latest_targets],
            ]
        self._draw_plot(canvas, (430, 10, 560, 270),
                        [self.eef_history_x, self.eef_history_y, self.eef_history_z],
                        tgt_axes,
                        [(0, 0, 255), (0, 220, 0), (255, 120, 0)],
                        ['X', 'Y', 'Z'])
        cv2.putText(canvas, 'EEF Position + Targets (dots)', (430, 296),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Right bottom: raw action bars
        if self.latest_raw_actions is not None:
            self._draw_action_bars(canvas, (430, 320, 560, 270), self.latest_raw_actions)
        cv2.putText(canvas, 'Raw Action pos (R=X G=Y B=Z per step)', (430, 606),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow('UMI Policy Bridge Debug', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit


def main():
    rclpy.init()
    node = PolicyVisualizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
