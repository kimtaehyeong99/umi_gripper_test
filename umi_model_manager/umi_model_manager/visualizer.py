#!/usr/bin/env python3
"""
UMI Policy Bridge Visualizer

Subscribes to debug topics and shows real-time visualization:
- Left: Camera observation image (what the policy sees)
- Right top: EEF position XYZ + target trajectory
- Right bottom: Raw action values per step

Usage:
  ros2 run umi_policy_bridge visualizer
"""

import collections
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float64MultiArray

import cv2


class PolicyVisualizer(Node):
    def __init__(self):
        super().__init__('policy_visualizer')

        # State
        self.latest_image = None
        self.latest_eef_pose = None
        self.latest_targets = None
        self.latest_raw_actions = None

        # History for plotting
        self.max_history = 200
        self.eef_history_x = collections.deque(maxlen=self.max_history)
        self.eef_history_y = collections.deque(maxlen=self.max_history)
        self.eef_history_z = collections.deque(maxlen=self.max_history)

        # Subscribers
        self.create_subscription(
            Image, '/umi_policy_bridge/debug/obs_image',
            self._image_cb, 5)
        self.create_subscription(
            PoseStamped, '/r_gripper_pose',
            self._eef_cb, 10)
        self.create_subscription(
            PoseArray, '/umi_policy_bridge/debug/target_poses',
            self._targets_cb, 5)
        self.create_subscription(
            Float64MultiArray, '/umi_policy_bridge/debug/raw_actions',
            self._actions_cb, 5)

        # Timer for rendering
        self.create_timer(0.1, self._render)  # 10Hz
        self.get_logger().info('Visualizer started. Waiting for data...')

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
        self.latest_targets = [
            (p.position.x, p.position.y, p.position.z) for p in msg.poses
        ]

    def _actions_cb(self, msg):
        data = np.array(msg.data)
        n_steps = len(data) // 10
        if n_steps > 0:
            self.latest_raw_actions = data.reshape(n_steps, 10)

    def _render(self):
        canvas_h, canvas_w = 600, 1000
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # ── Left panel: observation image (0-400 x 0-400) ──
        if self.latest_image is not None:
            img_resized = cv2.resize(self.latest_image, (400, 400))
            canvas[0:400, 0:400] = img_resized
        cv2.putText(canvas, 'Observation Image', (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Right top: EEF position history + targets (400-1000 x 0-300) ──
        plot_x, plot_y, plot_w, plot_h = 420, 10, 560, 260
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (40, 40, 40), -1)
        cv2.putText(canvas, 'EEF Position (XYZ) + Targets', (plot_x, plot_y + plot_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if len(self.eef_history_z) > 1:
            n = len(self.eef_history_x)
            # Plot each axis
            for data, color, label in [
                (self.eef_history_x, (0, 0, 255), 'X'),
                (self.eef_history_y, (0, 255, 0), 'Y'),
                (self.eef_history_z, (255, 100, 0), 'Z'),
            ]:
                arr = np.array(data)
                vmin, vmax = arr.min() - 0.02, arr.max() + 0.02
                if vmax - vmin < 0.01:
                    vmax = vmin + 0.01
                for i in range(1, n):
                    x1 = plot_x + int((i - 1) / self.max_history * plot_w)
                    x2 = plot_x + int(i / self.max_history * plot_w)
                    y1 = plot_y + plot_h - int((arr[i - 1] - vmin) / (vmax - vmin) * plot_h)
                    y2 = plot_y + plot_h - int((arr[i] - vmin) / (vmax - vmin) * plot_h)
                    y1 = max(plot_y, min(plot_y + plot_h, y1))
                    y2 = max(plot_y, min(plot_y + plot_h, y2))
                    cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

                # Label with latest value
                cv2.putText(canvas, f'{label}={arr[-1]:.3f}', (plot_x + plot_w - 100, plot_y + 15 + {'X': 0, 'Y': 15, 'Z': 30}[label]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Draw target positions as dots at the right edge
            if self.latest_targets:
                arr_x = np.array(self.eef_history_x)
                arr_y = np.array(self.eef_history_y)
                arr_z = np.array(self.eef_history_z)
                for axis_data, targets_axis, color in [
                    (arr_x, [t[0] for t in self.latest_targets], (0, 0, 255)),
                    (arr_y, [t[1] for t in self.latest_targets], (0, 255, 0)),
                    (arr_z, [t[2] for t in self.latest_targets], (255, 100, 0)),
                ]:
                    vmin, vmax = axis_data.min() - 0.02, axis_data.max() + 0.02
                    if vmax - vmin < 0.01:
                        vmax = vmin + 0.01
                    for j, tv in enumerate(targets_axis):
                        ty = plot_y + plot_h - int((tv - vmin) / (vmax - vmin) * plot_h)
                        ty = max(plot_y, min(plot_y + plot_h, ty))
                        tx = plot_x + plot_w - 5 + j * 3
                        if tx < plot_x + plot_w + 20:
                            cv2.circle(canvas, (min(tx, plot_x + plot_w - 1), ty), 2, color, -1)

        # ── Right bottom: Raw action values (400-1000 x 310-590) ──
        act_x, act_y, act_w, act_h = 420, 310, 560, 260
        cv2.rectangle(canvas, (act_x, act_y), (act_x + act_w, act_y + act_h), (40, 40, 40), -1)
        cv2.putText(canvas, 'Raw Action (pos3 per step)', (act_x, act_y + act_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if self.latest_raw_actions is not None:
            acts = self.latest_raw_actions
            n_steps = len(acts)
            bar_w = act_w // (n_steps * 3 + n_steps - 1)
            if bar_w < 2:
                bar_w = 2

            # Find scale
            pos_vals = acts[:, :3].flatten()
            max_abs = max(np.abs(pos_vals).max(), 0.01)

            for step_i in range(n_steps):
                for axis_j, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 100, 0)]):
                    val = acts[step_i, axis_j]
                    bar_idx = step_i * 4 + axis_j
                    bx = act_x + 10 + bar_idx * (bar_w + 1)
                    mid_y = act_y + act_h // 2
                    bar_height = int(val / max_abs * (act_h // 2 - 10))
                    bar_height = max(-act_h // 2 + 10, min(act_h // 2 - 10, bar_height))

                    if bar_height >= 0:
                        cv2.rectangle(canvas, (bx, mid_y - bar_height), (bx + bar_w, mid_y), color, -1)
                    else:
                        cv2.rectangle(canvas, (bx, mid_y), (bx + bar_w, mid_y - bar_height), color, -1)

                # Step label
                lx = act_x + 10 + step_i * 4 * (bar_w + 1)
                cv2.putText(canvas, str(step_i), (lx, act_y + act_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

            # Center line
            cv2.line(canvas, (act_x, act_y + act_h // 2), (act_x + act_w, act_y + act_h // 2), (80, 80, 80), 1)

            # Grip value
            cv2.putText(canvas, f'grip={acts[0, 9]:.3f}', (act_x + act_w - 100, act_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(canvas, f'max_abs={max_abs:.4f}', (act_x + act_w - 120, act_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # ── Bottom left: Current EEF text ──
        if self.latest_eef_pose:
            x, y, z = self.latest_eef_pose
            cv2.putText(canvas, f'EEF: x={x:.4f} y={y:.4f} z={z:.4f}',
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if self.latest_targets:
            t = self.latest_targets[0]
            cv2.putText(canvas, f'Tgt[0]: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f}',
                        (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

        cv2.imshow('UMI Policy Bridge Debug', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise SystemExit

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyVisualizer()
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
