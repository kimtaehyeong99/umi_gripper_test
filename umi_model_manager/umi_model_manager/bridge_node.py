#!/usr/bin/env python3
"""
UMI Policy Bridge Node

Bridges the UMI diffusion policy inference server (ZMQ) to the AI Worker robot (ROS2).
Collects observations, sends to inference server, converts actions, publishes delta poses.
"""

import threading
import time
import collections
from typing import Optional

import cv2
import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CompressedImage, JointState
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration

from umi_model_manager.pose_utils import (
    pose_to_mat, mat_to_pose, mat_to_pose10d, pose10d_to_mat,
    convert_pose_mat_rep, pose_stamped_to_pose6d,
    compute_delta_pose, rotvec_to_quaternion,
    rot6d_to_mat, mat_to_rot6d,
    build_T_link7_cam, apply_T_link7_cam, apply_T_cam_link7,
)


ObsFrame = collections.namedtuple('ObsFrame', ['timestamp', 'image', 'eef_pose', 'gripper_width'])


class UMIPolicyBridgeNode(Node):
    def __init__(self):
        super().__init__('umi_policy_bridge')

        # ── Parameters ──
        self.declare_parameter('zmq_host', '172.17.0.1')
        self.declare_parameter('zmq_port', 8766)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_rect_raw')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('eef_pose_topic', '/r_gripper_pose')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('gripper_joint_name', 'gripper_r_joint1')
        self.declare_parameter('delta_pose_topic', '/odom/camera_pose')
        self.declare_parameter('gripper_traj_topic',
                               '/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory')
        self.declare_parameter('image_size', [224, 224])
        self.declare_parameter('n_obs_steps', 2)
        self.declare_parameter('obs_down_sample_steps', 3)
        self.declare_parameter('obs_buffer_hz', 30.0)
        self.declare_parameter('n_action_steps', 8)
        self.declare_parameter('action_start_step', 1)  # Skip action[0] (≈identity)
        self.declare_parameter('action_hz', 20.0)
        self.declare_parameter('max_pos_delta', 0.05)
        self.declare_parameter('max_rot_delta', 0.5)
        self.declare_parameter('position_scale', 1.0)
        self.declare_parameter('gripper_width_to_joint_scale', 13.75)
        self.declare_parameter('gripper_width_to_joint_offset', 0.0)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('zmq_timeout_ms', 5000)
        self.declare_parameter('use_cam_frame', True)
        self.declare_parameter('delta_frame_id', 'odom')

        self.zmq_host = self.get_parameter('zmq_host').value
        self.zmq_port = self.get_parameter('zmq_port').value
        self.image_size = self.get_parameter('image_size').value
        self.n_obs_steps = self.get_parameter('n_obs_steps').value
        self.obs_down_sample_steps = self.get_parameter('obs_down_sample_steps').value
        self.n_action_steps = self.get_parameter('n_action_steps').value
        self.action_hz = self.get_parameter('action_hz').value
        self.max_pos_delta = self.get_parameter('max_pos_delta').value
        self.max_rot_delta = self.get_parameter('max_rot_delta').value
        self.position_scale = self.get_parameter('position_scale').value
        self.gripper_scale = self.get_parameter('gripper_width_to_joint_scale').value
        self.gripper_offset = self.get_parameter('gripper_width_to_joint_offset').value
        self.dry_run = self.get_parameter('dry_run').value
        self.zmq_timeout_ms = self.get_parameter('zmq_timeout_ms').value
        self.gripper_joint_name = self.get_parameter('gripper_joint_name').value

        # ── Coordinate frame calibration ──
        self.use_cam_frame = self.get_parameter('use_cam_frame').value
        if self.use_cam_frame:
            self.T_link7_cam = build_T_link7_cam()
            self.get_logger().info('Camera frame calibration ENABLED (link7 -> camera optical)')
        else:
            self.T_link7_cam = np.eye(4)
            self.get_logger().info('Camera frame calibration DISABLED (raw link7 frame)')

        # ── State ──
        self.obs_buffer: list[ObsFrame] = []
        self.obs_buffer_lock = threading.Lock()
        self.max_buffer_size = 60  # ~2 seconds at 30Hz

        self.latest_image: Optional[np.ndarray] = None
        self.latest_eef_pose: Optional[np.ndarray] = None  # [x,y,z,rx,ry,rz] in link7 frame
        self.latest_gripper_width: float = 0.0
        self.data_lock = threading.Lock()

        self.episode_start_pose: Optional[np.ndarray] = None  # [x,y,z,rx,ry,rz] in cam frame
        self.episode_start_eef_link7: Optional[np.ndarray] = None  # seed pose for controller
        self.running = False
        self.inference_thread: Optional[threading.Thread] = None

        # ── Subscribers ──
        cb_group = ReentrantCallbackGroup()

        self.use_compressed = self.get_parameter('use_compressed').value
        camera_topic = self.get_parameter('camera_topic').value
        if self.use_compressed:
            # Subscribe to compressed topic
            compressed_topic = camera_topic + '/compressed'
            self.camera_sub = self.create_subscription(
                CompressedImage,
                compressed_topic,
                self._compressed_camera_callback,
                10,
                callback_group=cb_group)
            self.get_logger().info(f'Camera: {compressed_topic} (compressed)')
        else:
            self.camera_sub = self.create_subscription(
                Image,
                camera_topic,
                self._camera_callback,
                10,
                callback_group=cb_group)
            self.get_logger().info(f'Camera: {camera_topic} (raw)')

        self.eef_pose_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter('eef_pose_topic').value,
            self._eef_pose_callback,
            10,
            callback_group=cb_group)

        self.joint_states_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self._joint_states_callback,
            10,
            callback_group=cb_group)

        # ── Publishers ──
        self.delta_pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('delta_pose_topic').value,
            10)

        self.gripper_traj_pub = self.create_publisher(
            JointTrajectory,
            self.get_parameter('gripper_traj_topic').value,
            10)

        # ── Debug publishers ──
        self.debug_image_pub = self.create_publisher(
            Image, '~/debug/obs_image', 5)
        self.debug_targets_pub = self.create_publisher(
            PoseArray, '~/debug/target_poses', 5)
        self.debug_actions_pub = self.create_publisher(
            Float64MultiArray, '~/debug/raw_actions', 5)

        # ── Observation buffer timer ──
        obs_period = 1.0 / self.get_parameter('obs_buffer_hz').value
        self.obs_timer = self.create_timer(obs_period, self._obs_timer_callback, callback_group=cb_group)

        # ── Episode control service ──
        self.start_srv = self.create_service(Trigger, '~/start_episode', self._start_episode_cb)
        self.stop_srv = self.create_service(Trigger, '~/stop_episode', self._stop_episode_cb)

        # ── ZMQ setup ──
        self.zmq_context = zmq.Context()
        self.zmq_socket = None

        self.get_logger().info(
            f'UMI Policy Bridge initialized (dry_run={self.dry_run})')
        self.get_logger().info(
            f'  ZMQ server: {self.zmq_host}:{self.zmq_port}')
        self.get_logger().info(
            f'  action_hz={self.action_hz}, n_action_steps={self.n_action_steps}')

    # ================================================================
    # Subscriber callbacks
    # ================================================================

    def _camera_callback(self, msg: Image):
        """Convert ROS Image to numpy BGR array."""
        try:
            # Handle common encodings without cv_bridge dependency issues
            h, w = msg.height, msg.width
            if msg.encoding in ('rgb8', 'bgr8'):
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                if msg.encoding == 'bgr8':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif msg.encoding == 'rgba8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            else:
                self.get_logger().warn(f'Unsupported image encoding: {msg.encoding}', throttle_duration_sec=5.0)
                return
            with self.data_lock:
                self.latest_image = img
        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}', throttle_duration_sec=5.0)

    def _compressed_camera_callback(self, msg: CompressedImage):
        """Decode compressed image (JPEG/PNG) to numpy RGB array."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with self.data_lock:
                self.latest_image = img
        except Exception as e:
            self.get_logger().error(f'Compressed camera error: {e}', throttle_duration_sec=5.0)

    def _eef_pose_callback(self, msg: PoseStamped):
        """Convert PoseStamped to [x,y,z,rx,ry,rz]."""
        pose6d = pose_stamped_to_pose6d(msg)
        with self.data_lock:
            self.latest_eef_pose = pose6d

    def _joint_states_callback(self, msg: JointState):
        """Extract gripper joint position."""
        if self.gripper_joint_name in msg.name:
            idx = msg.name.index(self.gripper_joint_name)
            joint_pos = msg.position[idx]
            # Convert joint position back to gripper width (meters)
            width = (joint_pos - self.gripper_offset) / self.gripper_scale
            with self.data_lock:
                self.latest_gripper_width = max(0.0, width)

    # ================================================================
    # Observation buffer
    # ================================================================

    def _obs_timer_callback(self):
        """Buffer latest observation at fixed rate."""
        with self.data_lock:
            image = self.latest_image
            pose = self.latest_eef_pose
            gripper = self.latest_gripper_width

        if pose is None:
            return  # Wait for first EEF pose

        # Skip if camera not available yet
        if image is None:
            return

        # Resize image
        if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]),
                               interpolation=cv2.INTER_LINEAR)

        # Convert EEF pose to camera optical frame for consistency with training data
        if self.use_cam_frame:
            cam_pose = apply_T_link7_cam(pose, self.T_link7_cam)
        else:
            cam_pose = pose.copy()

        frame = ObsFrame(
            timestamp=time.monotonic(),
            image=image.copy(),
            eef_pose=cam_pose,
            gripper_width=gripper,
        )

        with self.obs_buffer_lock:
            self.obs_buffer.append(frame)
            if len(self.obs_buffer) > self.max_buffer_size:
                self.obs_buffer = self.obs_buffer[-self.max_buffer_size:]

    # ================================================================
    # Episode control
    # ================================================================

    def _start_episode_cb(self, request, response):
        if self.running:
            response.success = False
            response.message = 'Episode already running'
            return response

        with self.data_lock:
            if self.latest_eef_pose is None:
                response.success = False
                response.message = 'No EEF pose received yet'
                return response
            # Store episode start pose in camera frame
            link7_pose = self.latest_eef_pose.copy()
            if self.use_cam_frame:
                self.episode_start_pose = apply_T_link7_cam(link7_pose, self.T_link7_cam)
            else:
                self.episode_start_pose = link7_pose

        with self.obs_buffer_lock:
            self.obs_buffer.clear()

        # Store seed pose (link7 frame) - controller uses this as base for relative poses
        with self.data_lock:
            self.episode_start_eef_link7 = self.latest_eef_pose.copy()

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

        self.get_logger().info('Episode started')
        response.success = True
        response.message = 'Episode started'
        return response

    def _stop_episode_cb(self, request, response):
        self.running = False
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=3.0)
            self.inference_thread = None
        self.get_logger().info('Episode stopped')
        response.success = True
        response.message = 'Episode stopped'
        return response

    # ================================================================
    # ZMQ inference client
    # ================================================================

    def _connect_zmq(self):
        if self.zmq_socket is not None:
            self.zmq_socket.close()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, self.zmq_timeout_ms)
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, self.zmq_timeout_ms)
        self.zmq_socket.connect(f'tcp://{self.zmq_host}:{self.zmq_port}')
        self.get_logger().info(f'ZMQ connected to {self.zmq_host}:{self.zmq_port}')

    def _zmq_predict(self, obs_dict_np: dict) -> Optional[np.ndarray]:
        """Send obs_dict to server, receive action."""
        try:
            if self.zmq_socket is None:
                self._connect_zmq()
            self.zmq_socket.send_pyobj(obs_dict_np)
            result = self.zmq_socket.recv_pyobj()
            if isinstance(result, str):
                self.get_logger().error(f'Server error: {result}')
                return None
            return result  # [16, 10]
        except zmq.ZMQError as e:
            self.get_logger().error(f'ZMQ error: {e}')
            self._connect_zmq()
            return None

    # ================================================================
    # Build observation dict
    # ================================================================

    def _build_obs_dict(self) -> Optional[dict]:
        """Build UMI-format observation dict from buffer."""
        with self.obs_buffer_lock:
            buf = list(self.obs_buffer)

        # Need at least (down_sample_steps + 1) frames for 2 observation steps
        min_frames = self.obs_down_sample_steps + 1
        if len(buf) < min_frames:
            return None

        # Select 2 frames: latest and (down_sample_steps) earlier
        frames = [buf[-(self.obs_down_sample_steps + 1)], buf[-1]]

        # Images: [T, H, W, 3] uint8 -> [T, 3, H, W] float32
        images = np.stack([f.image for f in frames])  # [T, H, W, 3]
        images_tchw = np.moveaxis(images, -1, 1).astype(np.float32) / 255.0  # [T, 3, H, W]

        # EEF poses: [x,y,z,rx,ry,rz] for each frame
        eef_poses = np.stack([f.eef_pose for f in frames])  # [T, 6]

        # Gripper width: [T, 1]
        gripper_widths = np.array([[f.gripper_width] for f in frames], dtype=np.float32)

        # ── Convert to relative representation ──
        # Build env_obs format for get_real_umi_obs_dict logic
        eef_pos = eef_poses[:, :3]      # [T, 3]
        eef_rot_aa = eef_poses[:, 3:]   # [T, 3] axis-angle

        # Convert to 4x4 pose matrices
        pose_mats = pose_to_mat(eef_poses)  # [T, 4, 4]

        # Relative pose: inv(last_obs) @ pose
        obs_pose_mat = convert_pose_mat_rep(
            pose_mats,
            base_pose_mat=pose_mats[-1],
            pose_rep='relative',
            backward=False)

        # Convert to 10D (pos3 + rot6d)
        obs_pose10d = mat_to_pose10d(obs_pose_mat)  # [T, 10]
        rel_eef_pos = obs_pose10d[:, :3].astype(np.float32)
        rel_eef_rot6d = obs_pose10d[:, 3:].astype(np.float32)  # [T, 6] -> goes to shape [6]

        # Rotation wrt episode start
        if self.episode_start_pose is not None:
            start_pose_mat = pose_to_mat(self.episode_start_pose)
            wrt_start_mat = convert_pose_mat_rep(
                pose_mats,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            wrt_start_10d = mat_to_pose10d(wrt_start_mat)
            wrt_start_rot6d = wrt_start_10d[:, 3:].astype(np.float32)
        else:
            # Identity rotation in 6D: first two columns of I flattened row-major
            wrt_start_rot6d = np.tile(
                np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (len(frames), 1))

        obs_dict = {
            'camera0_rgb': images_tchw,                                    # [T, 3, 224, 224]
            'robot0_eef_pos': rel_eef_pos,                                 # [T, 3]
            'robot0_eef_rot_axis_angle': rel_eef_rot6d,                    # [T, 6]
            'robot0_eef_rot_axis_angle_wrt_start': wrt_start_rot6d,        # [T, 6]
            'robot0_gripper_width': gripper_widths,                        # [T, 1]
        }

        return obs_dict

    # ================================================================
    # Convert policy actions to robot commands
    # ================================================================

    def _convert_actions(self, action: np.ndarray, last_obs_pose6d_cam: np.ndarray):
        """
        Convert policy action [N, 10] (relative, in camera frame) to absolute
        target poses in base_link/link7 frame.

        Args:
            action: [N, 10] relative actions from policy
            last_obs_pose6d_cam: last observation pose in camera frame

        Returns:
            target_poses: list of [x,y,z,rx,ry,rz] in base_link frame (link7)
            gripper_widths: list of float (meters)
        """
        n_steps = min(self.n_action_steps, len(action))
        last_obs_mat = pose_to_mat(last_obs_pose6d_cam)

        target_poses = []
        gripper_widths = []

        for i in range(n_steps):
            # Extract 10D action
            action_pose10d = action[i, :9]
            action_grip = action[i, 9]

            # Convert 10D relative action to 4x4 matrix
            action_mat = pose10d_to_mat(action_pose10d)

            # Relative -> absolute in camera frame: abs_cam = base_cam @ relative
            abs_mat_cam = convert_pose_mat_rep(
                action_mat,
                base_pose_mat=last_obs_mat,
                pose_rep='relative',
                backward=True)

            # Convert to pose6d (still in camera frame conceptually)
            abs_pose_cam = mat_to_pose(abs_mat_cam)

            # Apply position scale
            if self.position_scale != 1.0:
                delta_pos = abs_pose_cam[:3] - last_obs_pose6d_cam[:3]
                abs_pose_cam[:3] = last_obs_pose6d_cam[:3] + delta_pos * self.position_scale

            # Convert back from camera frame to link7/base_link frame
            if self.use_cam_frame:
                abs_pose_link7 = apply_T_cam_link7(abs_pose_cam, self.T_link7_cam)
            else:
                abs_pose_link7 = abs_pose_cam

            target_poses.append(abs_pose_link7)
            gripper_widths.append(float(action_grip))

        return target_poses, gripper_widths

    # ================================================================
    # Publish commands
    # ================================================================

    def _publish_delta_pose(self, delta_pose6d: np.ndarray):
        """Publish delta pose as PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('delta_frame_id').value

        # Clamp position delta
        pos_norm = np.linalg.norm(delta_pose6d[:3])
        if pos_norm > self.max_pos_delta:
            delta_pose6d[:3] = delta_pose6d[:3] * (self.max_pos_delta / pos_norm)

        # Clamp rotation delta
        rot_norm = np.linalg.norm(delta_pose6d[3:])
        if rot_norm > self.max_rot_delta:
            delta_pose6d[3:] = delta_pose6d[3:] * (self.max_rot_delta / rot_norm)

        msg.pose.position.x = float(delta_pose6d[0])
        msg.pose.position.y = float(delta_pose6d[1])
        msg.pose.position.z = float(delta_pose6d[2])

        quat = rotvec_to_quaternion(delta_pose6d[3:])  # [x,y,z,w]
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self.delta_pose_pub.publish(msg)

    def _publish_gripper(self, gripper_width: float):
        """Publish gripper command as JointTrajectory."""
        joint_pos = gripper_width * self.gripper_scale + self.gripper_offset
        joint_pos = max(0.0, min(1.1, joint_pos))  # Clamp to joint limits

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [self.gripper_joint_name]

        point = JointTrajectoryPoint()
        point.positions = [joint_pos]
        point.time_from_start = Duration(sec=0, nanosec=50_000_000)  # 50ms
        msg.points = [point]

        self.gripper_traj_pub.publish(msg)

    # ================================================================
    # Debug publishing
    # ================================================================

    def _publish_debug(self, obs_dict, action, target_poses, last_frame):
        """Publish debug data for visualization."""
        now = self.get_clock().now().to_msg()

        # 1. Observation image (latest, 224x224 RGB)
        try:
            # obs_dict camera0_rgb is [T, 3, H, W] float32 0-1, take last frame
            img_float = obs_dict['camera0_rgb'][-1]  # [3, 224, 224]
            img_uint8 = (np.moveaxis(img_float, 0, -1) * 255).astype(np.uint8)  # [224,224,3] RGB
            img_msg = Image()
            img_msg.header.stamp = now
            img_msg.height, img_msg.width = img_uint8.shape[:2]
            img_msg.encoding = 'rgb8'
            img_msg.step = img_uint8.shape[1] * 3
            img_msg.data = img_uint8.tobytes()
            self.debug_image_pub.publish(img_msg)
        except Exception:
            pass

        # 2. Raw action values [n_action_steps * 10] flattened
        try:
            n = min(self.n_action_steps, len(action))
            msg = Float64MultiArray()
            msg.data = action[:n].flatten().tolist()
            self.debug_actions_pub.publish(msg)
        except Exception:
            pass

        # 3. Target poses (in link7/base_link frame) as PoseArray
        try:
            pa = PoseArray()
            pa.header.stamp = now
            pa.header.frame_id = 'base_link'
            for tp in target_poses:
                p = Pose()
                p.position.x = float(tp[0])
                p.position.y = float(tp[1])
                p.position.z = float(tp[2])
                quat = rotvec_to_quaternion(tp[3:])
                p.orientation.x = float(quat[0])
                p.orientation.y = float(quat[1])
                p.orientation.z = float(quat[2])
                p.orientation.w = float(quat[3])
                pa.poses.append(p)
            self.debug_targets_pub.publish(pa)
        except Exception:
            pass

    # ================================================================
    # Main inference loop (runs in separate thread)
    # ================================================================

    def _inference_loop(self):
        """Main control loop: observe -> infer -> act."""
        self.get_logger().info('Inference loop started')
        self._connect_zmq()

        action_dt = 1.0 / self.action_hz

        while self.running and rclpy.ok():
            loop_start = time.monotonic()

            # 1. Build observation dict
            obs_dict = self._build_obs_dict()
            if obs_dict is None:
                self.get_logger().info('Waiting for observations...', throttle_duration_sec=2.0)
                time.sleep(0.1)
                continue

            if self.dry_run:
                self.get_logger().info(
                    f'[DRY RUN] obs_dict shapes: '
                    + ', '.join(f'{k}: {v.shape}' for k, v in obs_dict.items()),
                    throttle_duration_sec=2.0)
                time.sleep(0.5)
                continue

            # 2. Get last observation's absolute pose for action conversion
            with self.obs_buffer_lock:
                last_frame = self.obs_buffer[-1] if self.obs_buffer else None
            if last_frame is None:
                continue
            last_obs_pose6d = last_frame.eef_pose.copy()

            # 3. Inference
            inference_start = time.monotonic()
            action = self._zmq_predict(obs_dict)
            inference_time = time.monotonic() - inference_start

            if action is None:
                self.get_logger().warn('Inference failed, skipping cycle')
                time.sleep(0.1)
                continue

            self.get_logger().info(
                f'Inference: {inference_time:.3f}s, action shape: {action.shape}',
                throttle_duration_sec=1.0)

            # === DEBUG: Log raw policy output ===
            self.get_logger().info(
                f'[DEBUG] Raw action[0] pos={action[0,:3]}, grip={action[0,9]:.4f}',
                throttle_duration_sec=2.0)
            self.get_logger().info(
                f'[DEBUG] last_obs_cam={last_obs_pose6d[:3]}',
                throttle_duration_sec=2.0)

            # 4. Convert relative actions to absolute target poses (output in link7 frame)
            target_poses, gripper_widths = self._convert_actions(action, last_obs_pose6d)

            # === Publish debug data ===
            self._publish_debug(obs_dict, action, target_poses, last_frame)

            # 5. Execute actions sequentially
            with self.data_lock:
                current_eef = self.latest_eef_pose.copy() if self.latest_eef_pose is not None else None
            if current_eef is None:
                continue

            # === FILE DEBUG: write every cycle to /tmp/bridge_debug.txt ===
            try:
                with open('/tmp/bridge_debug.txt', 'w') as f:
                    f.write(f'=== Cycle at {time.monotonic():.3f} ===\n')
                    f.write(f'current_eef_link7: {current_eef}\n')
                    f.write(f'last_obs_cam: {last_obs_pose6d}\n')
                    f.write(f'raw_action[0]: {action[0]}\n')
                    f.write(f'raw_action[7]: {action[7]}\n')
                    for i, tp in enumerate(target_poses[:3]):
                        f.write(f'target[{i}]_link7: {tp}\n')
                    f.write(f'target[7]_link7: {target_poses[-1]}\n')
                    # Relative from seed (what controller receives)
                    seed = self.episode_start_eef_link7
                    f.write(f'seed_link7: {seed}\n')
                    rel0 = compute_delta_pose(seed, target_poses[0])
                    f.write(f'rel_from_seed[0] (sent to ctrl): {rel0[:3]}\n')
                    rel7 = compute_delta_pose(seed, target_poses[-1])
                    f.write(f'rel_from_seed[7] (sent to ctrl): {rel7[:3]}\n')
                    f.write(f'total_pos_from_seed: {np.linalg.norm(rel7[:3]):.6f} m\n')
                    f.write(f'publish_topic: {self.get_parameter("delta_pose_topic").value}\n')
                    f.write(f'frame_id: {self.get_parameter("delta_frame_id").value}\n')
                    f.write(f'use_cam_frame: {self.use_cam_frame}\n')
            except Exception:
                pass

            # Controller uses: goal = applyDeltaPose(seed_pose, received_pose)
            # So we must send the RELATIVE POSE from seed (episode start EEF)
            # to each target, NOT incremental deltas between consecutive targets.
            seed_pose = self.episode_start_eef_link7
            for i, (target, grip) in enumerate(zip(target_poses, gripper_widths)):
                if not self.running:
                    break

                step_start = time.monotonic()
                # Relative pose from seed to target (what controller expects)
                relative_from_seed = compute_delta_pose(seed_pose, target)
                self._publish_delta_pose(relative_from_seed)
                self._publish_gripper(grip)

                elapsed = time.monotonic() - step_start
                sleep_time = action_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_time = time.monotonic() - loop_start

        # Cleanup
        if self.zmq_socket is not None:
            self.zmq_socket.close()
            self.zmq_socket = None
        self.get_logger().info('Inference loop stopped')

    def destroy_node(self):
        self.running = False
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=3.0)
        if self.zmq_socket is not None:
            self.zmq_socket.close()
        self.zmq_context.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UMIPolicyBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
