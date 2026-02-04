#!/usr/bin/env python3
"""
UMI Data Pipeline - Optimized Bag Processor with Direct SLAM

Optimized version (v2.0) with:
- Direct ORB-SLAM3 API calls via pybind11 (no ros2 bag play overhead)
- In-memory processing (no intermediate disk I/O)
- Vectorized timestamp matching (O(N log M) instead of O(N*M))
- Compressed image decoding support

Usage:
    # Direct SLAM mode (default, recommended)
    python3 process_bag_with_slam.py --input data/raw/session_001 --output data/processed/session_001

    # With ROS2 SLAM visualization (slower)
    python3 process_bag_with_slam.py --input data/raw/session_001 --output data/processed/session_001 --use-ros2-slam
"""

import argparse
import os
import sys
import json
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
import h5py
import yaml
import time

# Rotation utilities for UMI format
from rotation_utils import quaternion_to_axis_angle

# UMI standard image size
UMI_IMAGE_SIZE = (224, 224)

# ROS2 bag reading
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    ROSBAGS_AVAILABLE = True
except ImportError:
    ROSBAGS_AVAILABLE = False
    typestore = None
    print("Warning: rosbags not installed. Install with: pip install rosbags")

# ROS2 for SLAM pose subscription
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    import threading
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    print("Warning: rclpy not available. ROS2 SLAM mode will not work.")

# Direct ORB-SLAM3 Python bindings
DIRECT_SLAM_AVAILABLE = False
orb_slam3_py = None
try:
    import sys
    # Add the installed pybind11 module path
    sys.path.insert(0, '/home/robotis-ai/umi_ws/install/ros2_orb_slam3/lib/ros2_orb_slam3')
    import orb_slam3_py
    DIRECT_SLAM_AVAILABLE = True
    print("ORB-SLAM3 Python bindings loaded successfully")
except ImportError as e:
    print(f"Warning: ORB-SLAM3 Python bindings not available: {e}")
    print("  Direct SLAM mode will not work. Use --use-ros2-slam or --no-slam")


@dataclass
class InMemoryDataStore:
    """Store extracted data in memory instead of disk for fast processing."""

    # Images stored in memory
    rgb_images: List[np.ndarray] = field(default_factory=list)
    depth_images: List[np.ndarray] = field(default_factory=list)

    # Timestamps
    rgb_timestamps: List[float] = field(default_factory=list)
    depth_timestamps: List[float] = field(default_factory=list)

    # Gripper data: (timestamp, width)
    gripper_obs: List[Tuple[float, float]] = field(default_factory=list)
    gripper_action: List[Tuple[float, float]] = field(default_factory=list)

    # Camera info
    camera_info: Optional[Dict] = None

    def add_rgb(self, timestamp: float, image: np.ndarray):
        """Add RGB image with timestamp."""
        self.rgb_timestamps.append(timestamp)
        self.rgb_images.append(image)

    def add_depth(self, timestamp: float, depth: np.ndarray):
        """Add depth image with timestamp."""
        self.depth_timestamps.append(timestamp)
        self.depth_images.append(depth)

    def add_gripper_obs(self, timestamp: float, width: float):
        """Add gripper observation (from /joint_states)."""
        self.gripper_obs.append((timestamp, width))

    def add_gripper_action(self, timestamp: float, width: float):
        """Add gripper action (from /commands)."""
        self.gripper_action.append((timestamp, width))

    def to_numpy_arrays(self) -> Dict[str, np.ndarray]:
        """Convert lists to numpy arrays for vectorized operations."""
        return {
            'rgb_timestamps': np.array(self.rgb_timestamps, dtype=np.float64),
            'depth_timestamps': np.array(self.depth_timestamps, dtype=np.float64),
            'gripper_obs_timestamps': np.array([g[0] for g in self.gripper_obs], dtype=np.float64),
            'gripper_obs_widths': np.array([g[1] for g in self.gripper_obs], dtype=np.float32),
            'gripper_action_timestamps': np.array([g[0] for g in self.gripper_action], dtype=np.float64),
            'gripper_action_widths': np.array([g[1] for g in self.gripper_action], dtype=np.float32),
        }

    @property
    def num_rgb_frames(self) -> int:
        return len(self.rgb_images)

    @property
    def num_depth_frames(self) -> int:
        return len(self.depth_images)

    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        total = 0
        for img in self.rgb_images:
            total += img.nbytes
        for img in self.depth_images:
            total += img.nbytes
        return total / (1024 * 1024)


class DirectSLAMProcessor:
    """
    Direct SLAM processing using pybind11 bindings.
    No ROS2 bag play, no topic overhead - just direct API calls.
    """

    def __init__(self, vocab_path: str, settings_path: str):
        """
        Initialize ORB-SLAM3 directly.

        Args:
            vocab_path: Path to ORB vocabulary file (.txt.bin)
            settings_path: Path to camera settings YAML file
        """
        if not DIRECT_SLAM_AVAILABLE:
            raise RuntimeError("ORB-SLAM3 Python bindings not available")

        print(f"\nInitializing Direct SLAM Processor...")
        print(f"  Vocabulary: {vocab_path}")
        print(f"  Settings: {settings_path}")

        # Set library path for runtime linking
        import os
        lib_path = '/home/robotis-ai/umi_ws/install/ros2_orb_slam3/lib'
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if lib_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}"

        self.slam = orb_slam3_py.ORBSLAM3(
            vocab_path,
            settings_path,
            False  # No viewer for batch processing
        )
        self.poses = []

    def process_frames(self, data_store: InMemoryDataStore,
                       max_depth_diff: float = 0.05) -> List[Dict]:
        """
        Process all RGB-D frames and return poses.

        Args:
            data_store: InMemoryDataStore with RGB and depth images
            max_depth_diff: Maximum timestamp difference for RGB-Depth matching

        Returns:
            List of pose dictionaries
        """
        print(f"\nProcessing {data_store.num_rgb_frames} frames with Direct SLAM...")
        start_time = time.time()

        # Vectorized RGB-Depth timestamp matching
        rgb_ts = np.array(data_store.rgb_timestamps, dtype=np.float64)
        depth_ts = np.array(data_store.depth_timestamps, dtype=np.float64)

        # Find nearest depth for each RGB frame
        if len(depth_ts) > 0:
            sorted_indices = np.argsort(depth_ts)
            sorted_ts = depth_ts[sorted_indices]
            indices = np.searchsorted(sorted_ts, rgb_ts)
            indices = np.clip(indices, 0, len(sorted_ts) - 1)

            # Check neighbors
            indices_left = np.clip(indices - 1, 0, len(sorted_ts) - 1)
            diff_right = np.abs(sorted_ts[indices] - rgb_ts)
            diff_left = np.abs(sorted_ts[indices_left] - rgb_ts)
            use_left = diff_left < diff_right
            depth_sorted_indices = np.where(use_left, indices_left, indices)
            depth_indices = sorted_indices[depth_sorted_indices]

            # Mark invalid matches
            min_diff = np.minimum(diff_left, diff_right)
            depth_indices = np.where(min_diff <= max_depth_diff, depth_indices, -1)
        else:
            depth_indices = np.full(len(rgb_ts), -1, dtype=np.int64)

        # Process each frame
        self.poses = []
        valid_count = 0
        skipped_count = 0

        for i in range(data_store.num_rgb_frames):
            rgb = data_store.rgb_images[i]
            timestamp = data_store.rgb_timestamps[i]

            # Get matched depth
            depth_idx = depth_indices[i]
            if depth_idx < 0:
                skipped_count += 1
                continue

            depth = data_store.depth_images[depth_idx]

            # Convert depth to float32 (keep mm units)
            # ORB-SLAM3 applies DepthMapFactor internally (1000.0 for mm->m)
            if depth.dtype == np.uint16:
                depth_float = depth.astype(np.float32)
            else:
                depth_float = depth.astype(np.float32)

            # Track with SLAM
            result = self.slam.track_rgbd(rgb, depth_float, timestamp)

            if result is not None:
                position, quaternion = result
                pose = {
                    'timestamp': timestamp,
                    'position': list(position),
                    'quaternion': list(quaternion),
                    'is_valid': True
                }
                valid_count += 1
            else:
                # Tracking failed - store placeholder
                pose = {
                    'timestamp': timestamp,
                    'position': [0.0, 0.0, 0.0],
                    'quaternion': [0.0, 0.0, 0.0, 1.0],
                    'is_valid': False
                }

            self.poses.append(pose)

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{data_store.num_rgb_frames} frames, "
                      f"valid: {valid_count}, tracking state: {self.slam.get_tracking_state()}")

        elapsed = time.time() - start_time
        print(f"\nDirect SLAM complete in {elapsed:.1f}s")
        print(f"  Total frames: {data_store.num_rgb_frames}")
        print(f"  Skipped (no depth): {skipped_count}")
        print(f"  Valid poses: {valid_count}")
        print(f"  Processing rate: {data_store.num_rgb_frames / elapsed:.1f} fps")

        return self.poses

    def shutdown(self):
        """Shutdown SLAM system."""
        if hasattr(self, 'slam') and self.slam is not None:
            self.slam.shutdown()
            self.slam = None


class OptimizedBagProcessor:
    """Optimized ROS2 bag processor for UMI dataset creation."""

    def __init__(self, config_path: Optional[str] = None,
                 use_ros2_slam: bool = False):
        """
        Initialize processor.

        Args:
            config_path: Path to configuration YAML
            use_ros2_slam: If True, use ros2 bag play + SLAM node (slower but with visualization)
        """
        self.config = self._load_config(config_path)
        self.use_ros2_slam = use_ros2_slam
        self.data_store = InMemoryDataStore()

        # Default paths for ORB-SLAM3
        self.vocab_path = os.path.expanduser(
            "~/umi_ws/src/umi_gripper_test/ros2_orb_slam3/orb_slam3/Vocabulary/ORBvoc.txt.bin"
        )
        self.settings_path = os.path.expanduser(
            "~/umi_ws/src/umi_gripper_test/ros2_orb_slam3/orb_slam3/config/RGBD/RealSense_D405.yaml"
        )

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from yaml file."""
        default_config = {
            'camera': {
                'rgb_topic': '/camera/camera/color/image_rect_raw/compressed',
                'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
                'camera_info_topic': '/camera/camera/color/camera_info',
                'fps': 30,
            },
            'gripper': {
                'action_topic': '/gripper_position_controller/commands',
                'observation_topic': '/joint_states',
                'joint_name': 'gripper',
                'min_position': 0.0,
                'max_position': 1.0,
                'min_width': 0.0,
                'max_width': 0.08,
            },
            'output': {
                'image_size': [224, 224],
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                for key in user_config:
                    if key in default_config:
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]

        return default_config

    # ==================== Compressed Image Decoders ====================

    def _decode_compressed_image(self, msg) -> Optional[np.ndarray]:
        """Decode ROS CompressedImage message to numpy array (BGR format for ORB-SLAM3)."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR format, same as C++ SLAM node
            return img
        except Exception as e:
            print(f"Error decoding compressed image: {e}")
            return None

    def _decode_compressed_depth(self, msg) -> Optional[np.ndarray]:
        """Decode ROS CompressedDepth message to numpy array."""
        try:
            # CompressedDepth format: first 12 bytes are header
            # (compression config: 4 bytes type, 4 bytes quantization, 4 bytes reserved)
            depth_header_size = 12
            raw_data = msg.data[depth_header_size:]

            # Decompress PNG
            np_arr = np.frombuffer(raw_data, np.uint8)
            depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if depth is None:
                # Try without header (some implementations skip header)
                np_arr = np.frombuffer(msg.data, np.uint8)
                depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            return depth
        except Exception as e:
            print(f"Error decoding compressed depth: {e}")
            return None

    def _decode_raw_image(self, msg, encoding: str = 'rgb8') -> Optional[np.ndarray]:
        """Decode ROS raw Image message to numpy array."""
        try:
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                print(f"Unknown image encoding: {msg.encoding}")
                return None
            return img
        except Exception as e:
            print(f"Error decoding raw image: {e}")
            return None

    def _decode_raw_depth(self, msg) -> Optional[np.ndarray]:
        """Decode ROS raw Depth Image message to numpy array."""
        try:
            if msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            elif msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                depth = (depth * 1000).astype(np.uint16)
            else:
                print(f"Unknown depth encoding: {msg.encoding}")
                return None
            return depth
        except Exception as e:
            print(f"Error decoding raw depth: {e}")
            return None

    def _decode_image_auto(self, msg, topic: str) -> Optional[np.ndarray]:
        """Auto-detect message type and decode image."""
        # Check if compressed based on topic name or message type
        if 'compressed' in topic.lower() or hasattr(msg, 'format'):
            return self._decode_compressed_image(msg)
        else:
            return self._decode_raw_image(msg)

    def _decode_depth_auto(self, msg, topic: str) -> Optional[np.ndarray]:
        """Auto-detect message type and decode depth."""
        if 'compressed' in topic.lower():
            return self._decode_compressed_depth(msg)
        else:
            return self._decode_raw_depth(msg)

    def _decode_gripper_command(self, msg) -> float:
        """Decode gripper command message (Float64MultiArray) to width in meters."""
        try:
            if hasattr(msg, 'data') and len(msg.data) > 0:
                position = msg.data[0]
            else:
                position = 0.0

            cfg = self.config['gripper']
            pos_range = cfg['max_position'] - cfg['min_position']
            width_range = cfg['max_width'] - cfg['min_width']

            if pos_range > 0:
                normalized = (position - cfg['min_position']) / pos_range
                width = cfg['min_width'] + normalized * width_range
            else:
                width = cfg['min_width']

            return float(width)
        except Exception as e:
            print(f"Error decoding gripper command: {e}")
            return 0.0

    def _decode_joint_states(self, msg) -> Optional[float]:
        """Decode JointState message to get gripper width in meters."""
        try:
            joint_name = self.config['gripper'].get('joint_name', 'gripper')

            if hasattr(msg, 'name') and hasattr(msg, 'position'):
                for i, name in enumerate(msg.name):
                    if joint_name in name:
                        position = msg.position[i]

                        cfg = self.config['gripper']
                        pos_range = cfg['max_position'] - cfg['min_position']
                        width_range = cfg['max_width'] - cfg['min_width']

                        if pos_range > 0:
                            normalized = (position - cfg['min_position']) / pos_range
                            width = cfg['min_width'] + normalized * width_range
                        else:
                            width = cfg['min_width']

                        return float(width)

            return None
        except Exception as e:
            print(f"Error decoding joint states: {e}")
            return None

    # ==================== In-Memory Data Extraction ====================

    def extract_data_to_memory(self, bag_path: str) -> InMemoryDataStore:
        """
        Extract all data from bag directly into memory.
        No disk I/O for images - massive speedup.
        """
        if not ROSBAGS_AVAILABLE:
            raise ImportError("rosbags package is required")

        print(f"Extracting data to memory from: {bag_path}")
        start_time = time.time()

        self.data_store = InMemoryDataStore()

        rgb_topic = self.config['camera']['rgb_topic']
        depth_topic = self.config['camera']['depth_topic']
        gripper_action_topic = self.config['gripper']['action_topic']
        gripper_obs_topic = self.config['gripper']['observation_topic']
        camera_info_topic = self.config['camera']['camera_info_topic']

        with Reader(bag_path) as reader:
            # Get available topics
            available_topics = {conn.topic for conn in reader.connections}
            print(f"Available topics in bag: {available_topics}")

            # Check which topics exist
            topics_to_read = []
            for topic in [rgb_topic, depth_topic, gripper_action_topic, gripper_obs_topic, camera_info_topic]:
                if topic in available_topics:
                    topics_to_read.append(topic)
                else:
                    print(f"  Warning: Topic {topic} not found in bag")

            print(f"Reading topics: {topics_to_read}")

            frame_count = 0
            for conn, timestamp, rawdata in reader.messages():
                topic = conn.topic
                ts_sec = timestamp / 1e9

                if topic == rgb_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    img = self._decode_image_auto(msg, topic)
                    if img is not None:
                        self.data_store.add_rgb(ts_sec, img)
                        frame_count += 1
                        if frame_count % 100 == 0:
                            print(f"  Extracted {frame_count} RGB frames...")

                elif topic == depth_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    depth = self._decode_depth_auto(msg, topic)
                    if depth is not None:
                        self.data_store.add_depth(ts_sec, depth)

                elif topic == gripper_action_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    width = self._decode_gripper_command(msg)
                    self.data_store.add_gripper_action(ts_sec, width)

                elif topic == gripper_obs_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    width = self._decode_joint_states(msg)
                    if width is not None:
                        self.data_store.add_gripper_obs(ts_sec, width)

                elif topic == camera_info_topic and self.data_store.camera_info is None:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    self.data_store.camera_info = {
                        'width': msg.width,
                        'height': msg.height,
                        'K': list(msg.k),
                        'D': list(msg.d),
                    }

        elapsed = time.time() - start_time
        mem_usage = self.data_store.get_memory_usage_mb()

        print(f"\nExtraction complete in {elapsed:.1f}s")
        print(f"  RGB frames: {self.data_store.num_rgb_frames}")
        print(f"  Depth frames: {self.data_store.num_depth_frames}")
        print(f"  Gripper observations: {len(self.data_store.gripper_obs)}")
        print(f"  Gripper actions: {len(self.data_store.gripper_action)}")
        print(f"  Memory usage: {mem_usage:.1f} MB")

        return self.data_store

    # ==================== Vectorized Timestamp Matching ====================

    @staticmethod
    def find_nearest_indices_vectorized(reference_ts: np.ndarray,
                                         target_ts: np.ndarray,
                                         max_diff: float = 0.1) -> np.ndarray:
        """
        Find nearest indices using vectorized numpy operations.
        O(N log M) instead of O(N*M).

        Args:
            reference_ts: Reference timestamps to match
            target_ts: Target timestamps to search in
            max_diff: Maximum allowed difference in seconds

        Returns:
            Array of indices (-1 for invalid matches)
        """
        if len(target_ts) == 0:
            return np.full(len(reference_ts), -1, dtype=np.int64)

        # Ensure sorted for searchsorted
        sorted_indices = np.argsort(target_ts)
        sorted_ts = target_ts[sorted_indices]

        # Find insertion points
        indices = np.searchsorted(sorted_ts, reference_ts)
        indices = np.clip(indices, 0, len(sorted_ts) - 1)

        # Check both neighbors (left and right)
        indices_left = np.clip(indices - 1, 0, len(sorted_ts) - 1)

        # Compute differences
        diff_right = np.abs(sorted_ts[indices] - reference_ts)
        diff_left = np.abs(sorted_ts[indices_left] - reference_ts)

        # Choose closer one
        use_left = diff_left < diff_right
        result_sorted = np.where(use_left, indices_left, indices)

        # Map back to original indices
        result = sorted_indices[result_sorted]

        # Mask invalid matches (beyond max_diff)
        min_diff = np.minimum(diff_left, diff_right)
        result = np.where(min_diff <= max_diff, result, -1)

        return result

    def synchronize_data_vectorized(self, poses: List[Dict]) -> Dict[str, Any]:
        """
        Vectorized synchronization of all data streams.
        Uses RGB timestamps as reference.
        """
        print("Synchronizing data (vectorized)...")
        start_time = time.time()

        arrays = self.data_store.to_numpy_arrays()
        rgb_ts = arrays['rgb_timestamps']
        n_frames = len(rgb_ts)

        if n_frames == 0:
            print("Error: No RGB frames found")
            return {}

        # Vectorized matching for all data streams
        depth_indices = self.find_nearest_indices_vectorized(
            rgb_ts, arrays['depth_timestamps']
        )
        gripper_obs_indices = self.find_nearest_indices_vectorized(
            rgb_ts, arrays['gripper_obs_timestamps']
        )
        gripper_action_indices = self.find_nearest_indices_vectorized(
            rgb_ts, arrays['gripper_action_timestamps']
        )

        # Pose matching
        if poses:
            pose_timestamps = np.array([p['timestamp'] for p in poses], dtype=np.float64)
            pose_indices = self.find_nearest_indices_vectorized(rgb_ts, pose_timestamps)
        else:
            pose_indices = np.full(n_frames, -1, dtype=np.int64)

        # Build synchronized data arrays
        gripper_obs_synced = np.zeros(n_frames, dtype=np.float32)
        gripper_action_synced = np.zeros(n_frames, dtype=np.float32)
        poses_synced = np.zeros((n_frames, 7), dtype=np.float32)
        poses_synced[:, 6] = 1.0  # Default quaternion w=1

        # Fill gripper observations
        valid_obs = gripper_obs_indices >= 0
        gripper_obs_synced[valid_obs] = arrays['gripper_obs_widths'][gripper_obs_indices[valid_obs]]

        # Fill gripper actions
        valid_action = gripper_action_indices >= 0
        gripper_action_synced[valid_action] = arrays['gripper_action_widths'][gripper_action_indices[valid_action]]

        # Fill poses
        valid_poses = pose_indices >= 0
        for i in np.where(valid_poses)[0]:
            pose = poses[pose_indices[i]]
            poses_synced[i, :3] = pose['position']
            poses_synced[i, 3:] = pose['quaternion']

        elapsed = time.time() - start_time
        print(f"Synchronization complete in {elapsed:.3f}s")
        print(f"  Valid depth matches: {np.sum(depth_indices >= 0)}/{n_frames}")
        print(f"  Valid gripper obs: {np.sum(valid_obs)}/{n_frames}")
        print(f"  Valid gripper action: {np.sum(valid_action)}/{n_frames}")
        print(f"  Valid poses: {np.sum(valid_poses)}/{n_frames}")

        return {
            'n_frames': n_frames,
            'timestamps': rgb_ts,
            'depth_indices': depth_indices,
            'gripper_obs': gripper_obs_synced,
            'gripper_action': gripper_action_synced,
            'poses': poses_synced,
            'valid_poses': valid_poses,
        }

    # ==================== SLAM Processing ====================

    def run_slam_with_bag_replay(self, bag_path: str) -> List[Dict]:
        """
        Run SLAM by playing back the bag file and collecting poses.
        Optimized version with smart initialization instead of hard sleep.
        """
        if not RCLPY_AVAILABLE:
            print("SLAM not available (rclpy not installed)")
            return []

        print("Starting SLAM with bag replay...")

        bag_path = str(Path(bag_path).resolve())
        print(f"Bag path: {bag_path}")

        # Initialize rclpy
        rclpy.init()

        class PoseCollector(Node):
            def __init__(self):
                super().__init__('pose_collector')
                self.poses = []
                self.subscription = self.create_subscription(
                    PoseStamped,
                    '/orb_slam3/camera_pose',
                    self.pose_callback,
                    10
                )
                self.get_logger().info('Pose collector initialized')

            def pose_callback(self, msg):
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                pose_data = {
                    'timestamp': timestamp,
                    'position': [
                        msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z
                    ],
                    'quaternion': [
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w
                    ]
                }
                self.poses.append(pose_data)
                if len(self.poses) % 100 == 0:
                    self.get_logger().info(f'Collected {len(self.poses)} poses')

        pose_collector = PoseCollector()
        env = os.environ.copy()

        # Get RGB/Depth topics from config
        rgb_topic = self.config['camera']['rgb_topic']
        depth_topic = self.config['camera']['depth_topic']

        # Start SLAM node (viewer disabled for subprocess compatibility)
        slam_cmd = f"ros2 run ros2_orb_slam3 rgbd_node_cpp --ros-args -p settings_name:=RealSense_D405 -p rgb_topic:={rgb_topic} -p depth_topic:={depth_topic} -p enable_viewer:=false"
        print(f"Starting SLAM: {slam_cmd}")

        slam_process = subprocess.Popen(
            slam_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid
        )

        # Smart SLAM initialization check instead of hard 5s sleep
        print("Waiting for SLAM initialization...")
        slam_ready = False
        max_wait = 15  # Maximum 15 seconds

        for i in range(max_wait * 2):  # Check every 0.5s
            if slam_process.poll() is not None:
                print(f"WARNING: SLAM process exited with code: {slam_process.poll()}")
                break

            # Check if SLAM node is publishing
            try:
                result = subprocess.run(
                    ['ros2', 'topic', 'list'],
                    capture_output=True, text=True, timeout=2
                )
                if '/orb_slam3/camera_pose' in result.stdout:
                    slam_ready = True
                    print(f"SLAM ready after {(i+1)*0.5:.1f}s")
                    break
            except:
                pass

            time.sleep(0.5)

        if not slam_ready:
            print("WARNING: SLAM may not be fully initialized, proceeding anyway...")

        # Start bag replay
        bag_cmd = f"ros2 bag play {bag_path} --rate 1.0"
        print(f"Starting bag replay: {bag_cmd}")

        bag_process = subprocess.Popen(
            bag_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid
        )

        # Collect poses while bag is playing
        print("Collecting poses from SLAM...")

        try:
            start_time = time.time()
            last_pose_count = 0

            while bag_process.poll() is None:
                rclpy.spin_once(pose_collector, timeout_sec=0.1)

                # Progress update
                if len(pose_collector.poses) > last_pose_count and len(pose_collector.poses) % 50 == 0:
                    print(f"  Collected {len(pose_collector.poses)} poses...")
                    last_pose_count = len(pose_collector.poses)

            elapsed = time.time() - start_time
            print(f"Bag finished after {elapsed:.1f}s")

            # Brief additional collection for remaining poses
            print("Collecting remaining poses...")
            for _ in range(50):  # 5 more seconds
                rclpy.spin_once(pose_collector, timeout_sec=0.1)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup using process groups
            print("Cleaning up SLAM processes...")

            def kill_process_group(proc, name):
                if proc.poll() is None:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                            proc.wait(timeout=2)
                        print(f"  {name} terminated")
                    except ProcessLookupError:
                        print(f"  {name} already terminated")
                    except Exception as e:
                        print(f"  Error terminating {name}: {e}")
                else:
                    print(f"  {name} already finished")

            kill_process_group(slam_process, "SLAM node")
            kill_process_group(bag_process, "Bag player")

            pose_collector.destroy_node()
            rclpy.shutdown()

        print(f"Collected {len(pose_collector.poses)} poses from SLAM")
        return pose_collector.poses

    def generate_placeholder_poses(self) -> List[Dict]:
        """Generate placeholder identity poses for all RGB frames."""
        poses = []
        for i, ts in enumerate(self.data_store.rgb_timestamps):
            poses.append({
                'timestamp': ts,
                'position': [0.0, 0.0, 0.0],
                'quaternion': [0.0, 0.0, 0.0, 1.0],
                'is_valid': False
            })
        print(f"Generated {len(poses)} placeholder poses")
        return poses

    # ==================== HDF5 Writing (UMI Format) ====================

    def save_to_hdf5_optimized(self, sync_data: Dict, output_path: str) -> None:
        """
        Save synchronized data to HDF5 in UMI-compatible format.

        UMI format requires:
        - camera0_rgb: [T, 224, 224, 3] uint8
        - robot0_eef_pos: [T, 3] float32 (position in meters)
        - robot0_eef_rot_axis_angle: [T, 3] float32 (axis-angle rotation)
        - robot0_gripper_width: [T, 1] float32 (gripper width in meters)
        - robot0_demo_start_pose: [T, 6] float32 (episode start pose)
        - robot0_demo_end_pose: [T, 6] float32 (episode end pose)
        - action: [T, 7] float32 ([pos3, rot3, gripper1])
        """
        print(f"Saving to HDF5 (UMI format): {output_path}")
        start_time = time.time()

        n_frames = sync_data['n_frames']
        if n_frames == 0:
            print("Error: No frames to save")
            return

        # Extract poses and convert quaternion to axis-angle
        poses_quat = sync_data['poses']  # [T, 7] = [x,y,z,qx,qy,qz,qw]
        positions = poses_quat[:, :3].astype(np.float32)  # [T, 3]

        # Convert quaternions to axis-angle
        quaternions = poses_quat[:, 3:7]  # [T, 4] = [qx,qy,qz,qw]
        axis_angles = np.zeros((n_frames, 3), dtype=np.float32)
        for i in range(n_frames):
            axis_angles[i] = quaternion_to_axis_angle(quaternions[i])

        # Gripper width - ensure shape [T, 1]
        gripper_widths = sync_data['gripper_obs'].reshape(-1, 1).astype(np.float32)

        # Create demo_start_pose and demo_end_pose [T, 6] = [pos3, rot3]
        start_pose = np.concatenate([positions[0], axis_angles[0]])  # [6]
        end_pose = np.concatenate([positions[-1], axis_angles[-1]])  # [6]
        demo_start_pose = np.tile(start_pose, (n_frames, 1)).astype(np.float64)  # [T, 6]
        demo_end_pose = np.tile(end_pose, (n_frames, 1)).astype(np.float64)  # [T, 6]

        # Create action array [T, 7] = [pos3, rot3, gripper1]
        # Action at time t = state at time t+1 (next frame target)
        action = np.zeros((n_frames, 7), dtype=np.float32)
        action[:-1, :3] = positions[1:]  # next position
        action[:-1, 3:6] = axis_angles[1:]  # next rotation
        action[:-1, 6:7] = gripper_widths[1:]  # next gripper
        action[-1] = action[-2]  # last frame copies previous

        with h5py.File(output_path, 'w') as f:
            # Create data group (UMI standard)
            data = f.create_group('data')

            # RGB images - resize to 224x224
            print("Writing images to HDF5 (resizing to 224x224)...")
            rgb_dataset = data.create_dataset(
                'camera0_rgb',
                shape=(n_frames, UMI_IMAGE_SIZE[0], UMI_IMAGE_SIZE[1], 3),
                dtype=np.uint8,
                compression='lzf'
            )

            depth_indices = sync_data['depth_indices']
            for i in range(n_frames):
                # Resize RGB to 224x224
                rgb_resized = cv2.resize(
                    self.data_store.rgb_images[i],
                    UMI_IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )
                rgb_dataset[i] = rgb_resized

                if (i + 1) % 200 == 0:
                    print(f"  Written {i + 1}/{n_frames} frames")

            # Store pose data in UMI format
            data.create_dataset('robot0_eef_pos', data=positions)
            data.create_dataset('robot0_eef_rot_axis_angle', data=axis_angles)
            data.create_dataset('robot0_gripper_width', data=gripper_widths)
            data.create_dataset('robot0_demo_start_pose', data=demo_start_pose)
            data.create_dataset('robot0_demo_end_pose', data=demo_end_pose)
            data.create_dataset('action', data=action)
            data.create_dataset('timestamps', data=sync_data['timestamps'])

            # Depth images (optional, for debugging)
            h, w = self.data_store.depth_images[0].shape[:2]
            depth_dataset = data.create_dataset(
                'depth_images',
                shape=(n_frames, h, w),
                dtype=np.uint16,
                compression='lzf'
            )
            for i in range(n_frames):
                if depth_indices[i] >= 0:
                    depth_dataset[i] = self.data_store.depth_images[depth_indices[i]]

            # Create meta group
            meta = f.create_group('meta')
            # Episode ends - cumulative indices where episodes end
            episode_ends = np.array([n_frames], dtype=np.int64)
            meta.create_dataset('episode_ends', data=episode_ends)

            # Additional metadata
            meta.attrs['num_episodes'] = 1
            meta.attrs['fps'] = self.config['camera']['fps']
            meta.attrs['num_frames'] = n_frames
            meta.attrs['image_size'] = UMI_IMAGE_SIZE

            if self.data_store.camera_info:
                meta.attrs['original_image_width'] = self.data_store.camera_info['width']
                meta.attrs['original_image_height'] = self.data_store.camera_info['height']
                meta.create_dataset('camera_K', data=np.array(self.data_store.camera_info['K']))
                if self.data_store.camera_info['D']:
                    meta.create_dataset('camera_D', data=np.array(self.data_store.camera_info['D']))

        elapsed = time.time() - start_time
        print(f"Saved {n_frames} frames to HDF5 (UMI format) in {elapsed:.1f}s")
        print(f"  camera0_rgb: [{n_frames}, {UMI_IMAGE_SIZE[0]}, {UMI_IMAGE_SIZE[1]}, 3]")
        print(f"  robot0_eef_pos: [{n_frames}, 3]")
        print(f"  robot0_eef_rot_axis_angle: [{n_frames}, 3]")
        print(f"  robot0_gripper_width: [{n_frames}, 1]")
        print(f"  action: [{n_frames}, 7]")

    def _run_direct_slam(self) -> List[Dict]:
        """Run Direct SLAM using pybind11 bindings."""
        print("Using Direct SLAM (pybind11 bindings)...")

        try:
            slam_processor = DirectSLAMProcessor(
                self.vocab_path,
                self.settings_path
            )
            poses = slam_processor.process_frames(self.data_store)
            slam_processor.shutdown()
            return poses
        except Exception as e:
            print(f"Direct SLAM failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to placeholder poses...")
            return self.generate_placeholder_poses()

    def save_trajectory_csv(self, poses: List[Dict], output_dir: str) -> None:
        """Save trajectory to CSV file."""
        traj_path = os.path.join(output_dir, 'camera_trajectory.csv')
        with open(traj_path, 'w') as f:
            f.write('frame_idx,timestamp,x,y,z,qx,qy,qz,qw,is_valid\n')
            for i, pose in enumerate(poses):
                is_valid = 1 if pose.get('is_valid', True) else 0
                f.write(f"{i},{pose['timestamp']:.6f},"
                       f"{pose['position'][0]:.6f},{pose['position'][1]:.6f},{pose['position'][2]:.6f},"
                       f"{pose['quaternion'][0]:.6f},{pose['quaternion'][1]:.6f},"
                       f"{pose['quaternion'][2]:.6f},{pose['quaternion'][3]:.6f},"
                       f"{is_valid}\n")
        print(f"Saved trajectory to {traj_path}")

    # ==================== Main Processing Pipeline ====================

    def process(self, input_bag: str, output_dir: str) -> str:
        """
        Main optimized processing pipeline.

        Args:
            input_bag: Path to ROS2 bag directory
            output_dir: Output directory for processed data

        Returns:
            Path to output HDF5 file
        """
        os.makedirs(output_dir, exist_ok=True)
        total_start = time.time()

        # Step 1: Extract data to memory
        print("\n" + "="*60)
        print("Step 1: Extracting data to memory")
        print("="*60)
        self.extract_data_to_memory(input_bag)

        # Step 2: Run SLAM
        print("\n" + "="*60)
        print("Step 2: Running SLAM")
        print("="*60)

        if self.use_ros2_slam:
            # Use ROS2 bag play + SLAM node (slower, but with visualization option)
            if RCLPY_AVAILABLE:
                poses = self.run_slam_with_bag_replay(input_bag)
                if len(poses) == 0:
                    print("ROS2 SLAM failed, falling back to Direct SLAM...")
                    if DIRECT_SLAM_AVAILABLE:
                        poses = self._run_direct_slam()
                    else:
                        raise RuntimeError("Both ROS2 SLAM and Direct SLAM failed")
            else:
                print("rclpy not available, using Direct SLAM...")
                if DIRECT_SLAM_AVAILABLE:
                    poses = self._run_direct_slam()
                else:
                    raise RuntimeError("No SLAM method available")
        else:
            # Default: Direct SLAM (fastest, recommended)
            if DIRECT_SLAM_AVAILABLE:
                poses = self._run_direct_slam()
            elif RCLPY_AVAILABLE:
                print("Direct SLAM not available, falling back to ROS2 SLAM...")
                poses = self.run_slam_with_bag_replay(input_bag)
                if len(poses) == 0:
                    raise RuntimeError("ROS2 SLAM failed and Direct SLAM not available")
            else:
                raise RuntimeError("No SLAM method available. Install ORB-SLAM3 Python bindings or rclpy.")

        # Save trajectory
        self.save_trajectory_csv(poses, output_dir)

        # Step 3: Synchronize data (vectorized)
        print("\n" + "="*60)
        print("Step 3: Synchronizing data (vectorized)")
        print("="*60)
        sync_data = self.synchronize_data_vectorized(poses)

        # Step 4: Save to HDF5
        print("\n" + "="*60)
        print("Step 4: Saving to HDF5")
        print("="*60)
        hdf5_path = os.path.join(output_dir, 'dataset.hdf5')
        self.save_to_hdf5_optimized(sync_data, hdf5_path)

        # Determine SLAM mode used
        slam_mode = 'ros2_slam' if self.use_ros2_slam else 'direct_slam'

        # Save metadata
        metadata = {
            'input_bag': input_bag,
            'num_frames': sync_data['n_frames'],
            'slam_mode': slam_mode,
            'valid_poses': int(np.sum(sync_data['valid_poses'])),
            'config': self.config
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        total_elapsed = time.time() - total_start
        print("\n" + "="*60)
        print(f"Processing complete in {total_elapsed:.1f}s")
        print("="*60)
        print(f"Output directory: {output_dir}")
        print(f"HDF5 file: {hdf5_path}")

        return hdf5_path


def main():
    parser = argparse.ArgumentParser(
        description='Optimized ROS2 bag processor with Direct SLAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct SLAM mode (default, fastest with real poses)
  python3 process_bag_with_slam.py -i data/raw/session_01 -o data/processed/session_01

  # With ROS2 SLAM visualization (slower)
  python3 process_bag_with_slam.py -i data/raw/session_01 -o data/processed/session_01 --use-ros2-slam
        """
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input ROS2 bag directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for processed data')
    parser.add_argument('--config', '-c', default=None,
                        help='Path to config yaml file')
    parser.add_argument('--use-ros2-slam', action='store_true',
                        help='Use ROS2 bag play + SLAM node (slower, for visualization)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input bag not found: {args.input}")
        sys.exit(1)

    processor = OptimizedBagProcessor(
        config_path=args.config,
        use_ros2_slam=args.use_ros2_slam
    )
    processor.process(args.input, args.output)


if __name__ == '__main__':
    main()
