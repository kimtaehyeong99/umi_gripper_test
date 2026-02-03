#!/usr/bin/env python3
"""
UMI Data Pipeline - Process Bag with SLAM

This script processes a ROS2 bag file:
1. Extracts RGB/Depth images and gripper data
2. Runs ORB-SLAM3 offline to get camera trajectory
3. Synchronizes all data
4. Saves to HDF5 format

Usage:
    python3 process_bag_with_slam.py --input data/raw/session_001 --output data/processed/session_001
"""

import argparse
import os
import sys
import json
import subprocess
import signal
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import h5py
import yaml

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
    import time
    import signal
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    print("Warning: rclpy not available. SLAM integration will use placeholder.")


class BagProcessor:
    """Process ROS2 bag files for UMI dataset creation."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

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
                'action_topic': '/gripper_position_controller/commands',  # Action (command)
                'observation_topic': '/joint_states',  # Observation (actual state)
                'joint_name': 'gripper',  # Joint name in JointState message
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
                # Merge configs
                for key in user_config:
                    if key in default_config:
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]

        return default_config

    def extract_data_from_bag(self, bag_path: str, output_dir: str) -> Dict:
        """Extract RGB, Depth, and Gripper data from ROS2 bag."""

        if not ROSBAGS_AVAILABLE:
            raise ImportError("rosbags package is required. Install with: pip install rosbags")

        print(f"Extracting data from: {bag_path}")

        # Create output directories
        rgb_dir = os.path.join(output_dir, 'rgb')
        depth_dir = os.path.join(output_dir, 'depth')
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        rgb_topic = self.config['camera']['rgb_topic']
        depth_topic = self.config['camera']['depth_topic']
        gripper_action_topic = self.config['gripper']['action_topic']
        gripper_obs_topic = self.config['gripper']['observation_topic']

        rgb_data = []
        depth_data = []
        gripper_action_data = []  # From commands (action)
        gripper_obs_data = []     # From joint_states (observation)
        camera_info = None

        with Reader(bag_path) as reader:
            # Get connections for our topics
            connections = {}
            for conn in reader.connections:
                if conn.topic in [rgb_topic, depth_topic, gripper_action_topic,
                                  gripper_obs_topic, self.config['camera']['camera_info_topic']]:
                    connections[conn.topic] = conn

            print(f"Found topics: {list(connections.keys())}")

            # Read messages
            for conn, timestamp, rawdata in reader.messages():
                topic = conn.topic
                ts_sec = timestamp / 1e9  # Convert to seconds

                if topic == rgb_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    img = self._decode_image(msg, 'rgb8')
                    if img is not None:
                        frame_idx = len(rgb_data)
                        img_path = os.path.join(rgb_dir, f'{frame_idx:06d}.png')
                        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        rgb_data.append({
                            'timestamp': ts_sec,
                            'path': img_path,
                            'frame_idx': frame_idx
                        })

                elif topic == depth_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    depth = self._decode_depth(msg)
                    if depth is not None:
                        frame_idx = len(depth_data)
                        depth_path = os.path.join(depth_dir, f'{frame_idx:06d}.png')
                        cv2.imwrite(depth_path, depth)
                        depth_data.append({
                            'timestamp': ts_sec,
                            'path': depth_path,
                            'frame_idx': frame_idx
                        })

                elif topic == gripper_action_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    width = self._decode_gripper_command(msg)
                    gripper_action_data.append({
                        'timestamp': ts_sec,
                        'width': width
                    })

                elif topic == gripper_obs_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    width = self._decode_joint_states(msg)
                    if width is not None:
                        gripper_obs_data.append({
                            'timestamp': ts_sec,
                            'width': width
                        })

                elif topic == self.config['camera']['camera_info_topic'] and camera_info is None:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    camera_info = {
                        'width': msg.width,
                        'height': msg.height,
                        'K': list(msg.k),
                        'D': list(msg.d),
                    }

        print(f"Extracted: {len(rgb_data)} RGB, {len(depth_data)} Depth")
        print(f"  Gripper: {len(gripper_obs_data)} observations, {len(gripper_action_data)} actions")

        return {
            'rgb': rgb_data,
            'depth': depth_data,
            'gripper_observation': gripper_obs_data,  # From /joint_states
            'gripper_action': gripper_action_data,    # From /gripper_position_controller/commands
            'camera_info': camera_info
        }

    def _decode_image(self, msg, encoding: str) -> Optional[np.ndarray]:
        """Decode ROS Image message to numpy array."""
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
            print(f"Error decoding image: {e}")
            return None

    def _decode_depth(self, msg) -> Optional[np.ndarray]:
        """Decode ROS Depth Image message to numpy array."""
        try:
            if msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            elif msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                depth = (depth * 1000).astype(np.uint16)  # Convert to mm
            else:
                print(f"Unknown depth encoding: {msg.encoding}")
                return None
            return depth
        except Exception as e:
            print(f"Error decoding depth: {e}")
            return None

    def _decode_gripper_command(self, msg) -> float:
        """Decode gripper command message (Float64MultiArray) to width in meters."""
        try:
            # Float64MultiArray - take first element
            if hasattr(msg, 'data') and len(msg.data) > 0:
                position = msg.data[0]
            else:
                position = 0.0

            # Convert position to width using calibration
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

            # Find the gripper joint in JointState message
            if hasattr(msg, 'name') and hasattr(msg, 'position'):
                for i, name in enumerate(msg.name):
                    if joint_name in name:
                        position = msg.position[i]

                        # Convert position to width using calibration
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

    def run_slam_offline(self, rgb_dir: str, depth_dir: str, output_dir: str,
                          rgb_data: List[Dict] = None, bag_path: str = None) -> List[Dict]:
        """
        Run ORB-SLAM3 offline on extracted images.

        This uses ros2 bag play + SLAM node approach.

        Args:
            rgb_dir: Directory containing RGB images
            depth_dir: Directory containing depth images
            output_dir: Output directory for trajectory file
            rgb_data: Optional list of RGB frame data with timestamps
            bag_path: Path to the ROS2 bag file for replay
        """
        print("Running offline SLAM...")

        # Check if we can run real SLAM
        if not RCLPY_AVAILABLE or bag_path is None:
            print("SLAM not available, using placeholder poses...")
            return self._generate_placeholder_trajectory(rgb_dir, rgb_data, output_dir)

        try:
            trajectory = self._run_slam_with_bag_replay(bag_path, rgb_data, output_dir)
            if len(trajectory) == 0:
                print("SLAM failed, falling back to placeholder poses...")
                return self._generate_placeholder_trajectory(rgb_dir, rgb_data, output_dir)
            return trajectory
        except Exception as e:
            print(f"SLAM error: {e}")
            print("Falling back to placeholder poses...")
            return self._generate_placeholder_trajectory(rgb_dir, rgb_data, output_dir)

    def _run_slam_with_bag_replay(self, bag_path: str, rgb_data: List[Dict],
                                   output_dir: str) -> List[Dict]:
        """
        Run SLAM by playing back the bag file and collecting poses.
        """
        print("Starting SLAM with bag replay...")

        # Resolve absolute path for bag file
        bag_path = str(Path(bag_path).resolve())
        print(f"Bag path (absolute): {bag_path}")

        collected_poses = []
        slam_ready = threading.Event()
        bag_finished = threading.Event()

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

        # Prepare environment with ROS2 variables
        env = os.environ.copy()
        print(f"ROS_DOMAIN_ID: {env.get('ROS_DOMAIN_ID', 'not set')}")
        print(f"RMW_IMPLEMENTATION: {env.get('RMW_IMPLEMENTATION', 'not set')}")

        # Start SLAM node in subprocess (viewer MUST be disabled for subprocess compatibility)
        slam_cmd = "ros2 run ros2_orb_slam3 rgbd_node_cpp --ros-args -p settings_name:=RealSense_D405 -p rgb_topic:=/camera/camera/color/image_rect_raw/compressed -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw/compressedDepth -p enable_viewer:=false"
        print(f"Starting SLAM node: {slam_cmd}")
        slam_process = subprocess.Popen(
            slam_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid  # Create new process group for proper cleanup
        )

        # Wait for SLAM to initialize
        print("Waiting for SLAM node to initialize (5 seconds)...")
        time.sleep(5)

        # Check if SLAM started
        slam_poll = slam_process.poll()
        if slam_poll is not None:
            print(f"WARNING: SLAM process exited early with code: {slam_poll}")
            _, stderr = slam_process.communicate()
            if stderr:
                print(f"SLAM stderr: {stderr.decode()[:500]}")

        # Start bag play in subprocess
        bag_cmd = f"ros2 bag play {bag_path} --rate 1.0"
        print(f"Starting bag replay: {bag_cmd}")
        bag_process = subprocess.Popen(
            bag_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid  # Create new process group for proper cleanup
        )

        # Spin ROS node to collect poses while bag is playing
        print("Collecting poses from SLAM...")

        # Check if bag process started successfully
        time.sleep(1)
        bag_poll = bag_process.poll()
        if bag_poll is not None:
            print(f"WARNING: Bag process exited early with code: {bag_poll}")
            stdout, stderr = bag_process.communicate()
            if stderr:
                print(f"Bag stderr: {stderr.decode()}")

        try:
            start_time = time.time()
            while bag_process.poll() is None:
                rclpy.spin_once(pose_collector, timeout_sec=0.1)
                if len(pose_collector.poses) > 0 and len(pose_collector.poses) % 50 == 0:
                    print(f"  Collected {len(pose_collector.poses)} poses so far...")

            elapsed = time.time() - start_time
            print(f"Bag finished after {elapsed:.1f}s")

            # Continue spinning briefly to catch any remaining poses
            print("Waiting for SLAM to finish processing...")
            for _ in range(100):  # 10 more seconds
                rclpy.spin_once(pose_collector, timeout_sec=0.1)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup - kill entire process groups to ensure no orphan processes
            print("Cleaning up SLAM processes...")

            def kill_process_group(proc, name):
                """Kill process and all its children using process group."""
                if proc.poll() is None:  # Still running
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

        collected_poses = pose_collector.poses
        print(f"Collected {len(collected_poses)} poses from SLAM")

        # Match poses to RGB frames
        trajectory = self._match_poses_to_frames(collected_poses, rgb_data)

        # Save trajectory to CSV
        self._save_trajectory_csv(trajectory, output_dir)

        return trajectory

    def _generate_placeholder_trajectory(self, rgb_dir: str, rgb_data: List[Dict],
                                          output_dir: str) -> List[Dict]:
        """Generate placeholder identity poses."""
        rgb_files = sorted(Path(rgb_dir).glob('*.png'))
        trajectory = []

        for i, rgb_path in enumerate(rgb_files):
            if rgb_data and i < len(rgb_data):
                timestamp = rgb_data[i]['timestamp']
            else:
                timestamp = i / 30.0

            trajectory.append({
                'frame_idx': i,
                'timestamp': timestamp,
                'position': [0.0, 0.0, 0.0],
                'quaternion': [0.0, 0.0, 0.0, 1.0],
                'is_valid': True
            })

        print(f"Generated {len(trajectory)} placeholder pose estimates")
        self._save_trajectory_csv(trajectory, output_dir)
        return trajectory

    def _match_poses_to_frames(self, collected_poses: List[Dict],
                                rgb_data: List[Dict]) -> List[Dict]:
        """Match collected SLAM poses to RGB frame timestamps."""
        if not collected_poses or not rgb_data:
            return []

        trajectory = []
        pose_timestamps = np.array([p['timestamp'] for p in collected_poses])

        for i, rgb in enumerate(rgb_data):
            rgb_ts = rgb['timestamp']

            # Find closest pose
            diffs = np.abs(pose_timestamps - rgb_ts)
            min_idx = np.argmin(diffs)
            min_diff = diffs[min_idx]

            # Accept if within 100ms
            if min_diff < 0.1:
                pose = collected_poses[min_idx]
                trajectory.append({
                    'frame_idx': i,
                    'timestamp': rgb_ts,
                    'position': pose['position'],
                    'quaternion': pose['quaternion'],
                    'is_valid': True
                })
            else:
                # No matching pose found, use identity
                trajectory.append({
                    'frame_idx': i,
                    'timestamp': rgb_ts,
                    'position': [0.0, 0.0, 0.0],
                    'quaternion': [0.0, 0.0, 0.0, 1.0],
                    'is_valid': False
                })

        valid_count = sum(1 for t in trajectory if t['is_valid'])
        print(f"Matched {valid_count}/{len(trajectory)} frames with valid poses")

        return trajectory

    def _save_trajectory_csv(self, trajectory: List[Dict], output_dir: str) -> None:
        """Save trajectory to CSV file."""
        traj_path = os.path.join(output_dir, 'camera_trajectory.csv')
        with open(traj_path, 'w') as f:
            f.write('frame_idx,timestamp,x,y,z,qx,qy,qz,qw,is_valid\n')
            for pose in trajectory:
                f.write(f"{pose['frame_idx']},{pose['timestamp']:.6f},"
                       f"{pose['position'][0]:.6f},{pose['position'][1]:.6f},{pose['position'][2]:.6f},"
                       f"{pose['quaternion'][0]:.6f},{pose['quaternion'][1]:.6f},"
                       f"{pose['quaternion'][2]:.6f},{pose['quaternion'][3]:.6f},"
                       f"{1 if pose['is_valid'] else 0}\n")
        print(f"Saved trajectory to {traj_path}")

    def synchronize_data(self, extracted_data: Dict, trajectory: List[Dict]) -> List[Dict]:
        """Synchronize RGB, Depth, Gripper, and Pose data by timestamp."""

        print("Synchronizing data...")

        rgb_data = extracted_data['rgb']
        depth_data = extracted_data['depth']
        gripper_obs_data = extracted_data['gripper_observation']  # From /joint_states
        gripper_action_data = extracted_data['gripper_action']    # From /gripper_position_controller/commands

        if len(rgb_data) == 0:
            print("Error: No RGB data found")
            return []

        # Use RGB timestamps as reference
        synced_frames = []

        for i, rgb in enumerate(rgb_data):
            rgb_ts = rgb['timestamp']

            # Find closest depth
            depth_idx = self._find_closest_timestamp(
                rgb_ts, [d['timestamp'] for d in depth_data]
            )

            # Find closest gripper observation (from /joint_states)
            gripper_obs_idx = self._find_closest_timestamp(
                rgb_ts, [g['timestamp'] for g in gripper_obs_data]
            )

            # Find closest gripper action (from /gripper_position_controller/commands)
            gripper_action_idx = self._find_closest_timestamp(
                rgb_ts, [g['timestamp'] for g in gripper_action_data]
            )

            # Find closest pose
            pose_idx = self._find_closest_timestamp(
                rgb_ts, [p['timestamp'] for p in trajectory]
            )

            synced_frame = {
                'frame_idx': i,
                'timestamp': rgb_ts,
                'rgb_path': rgb['path'],
                'depth_path': depth_data[depth_idx]['path'] if depth_idx is not None else None,
                'gripper_observation': gripper_obs_data[gripper_obs_idx]['width'] if gripper_obs_idx is not None else 0.0,
                'gripper_action': gripper_action_data[gripper_action_idx]['width'] if gripper_action_idx is not None else 0.0,
                'pose': trajectory[pose_idx] if pose_idx is not None else None
            }
            synced_frames.append(synced_frame)

        print(f"Synchronized {len(synced_frames)} frames")
        return synced_frames

    def _find_closest_timestamp(self, target_ts: float, timestamps: List[float],
                                max_diff: float = 0.1) -> Optional[int]:
        """Find index of closest timestamp within max_diff seconds."""
        if len(timestamps) == 0:
            return None

        diffs = [abs(ts - target_ts) for ts in timestamps]
        min_idx = np.argmin(diffs)

        if diffs[min_idx] <= max_diff:
            return min_idx
        return None

    def save_to_hdf5(self, synced_frames: List[Dict], camera_info: Dict,
                     output_path: str) -> None:
        """Save synchronized data to HDF5 format."""

        print(f"Saving to HDF5: {output_path}")

        if len(synced_frames) == 0:
            print("Error: No frames to save")
            return

        with h5py.File(output_path, 'w') as f:
            # Create episode group
            episode = f.create_group('episode_0000')

            # Prepare arrays
            n_frames = len(synced_frames)
            timestamps = np.array([frame['timestamp'] for frame in synced_frames])
            gripper_observations = np.array([frame['gripper_observation'] for frame in synced_frames])
            gripper_actions = np.array([frame['gripper_action'] for frame in synced_frames])

            # Poses
            poses = np.zeros((n_frames, 7), dtype=np.float32)
            for i, frame in enumerate(synced_frames):
                if frame['pose'] is not None:
                    poses[i, :3] = frame['pose']['position']
                    poses[i, 3:] = frame['pose']['quaternion']

            # Read and store images
            first_rgb = cv2.imread(synced_frames[0]['rgb_path'])
            h, w = first_rgb.shape[:2]

            rgb_images = episode.create_dataset(
                'rgb_images',
                shape=(n_frames, h, w, 3),
                dtype=np.uint8,
                compression='gzip'
            )

            depth_images = episode.create_dataset(
                'depth_images',
                shape=(n_frames, h, w),
                dtype=np.uint16,
                compression='gzip'
            )

            for i, frame in enumerate(synced_frames):
                # RGB
                rgb = cv2.imread(frame['rgb_path'])
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_images[i] = rgb

                # Depth
                if frame['depth_path'] and os.path.exists(frame['depth_path']):
                    depth = cv2.imread(frame['depth_path'], cv2.IMREAD_UNCHANGED)
                    depth_images[i] = depth

                if (i + 1) % 100 == 0:
                    print(f"  Saved {i + 1}/{n_frames} frames")

            # Store other data
            episode.create_dataset('timestamps', data=timestamps)
            episode.create_dataset('camera_pose', data=poses)
            episode.create_dataset('gripper_width', data=gripper_observations)  # Observation from /joint_states
            episode.create_dataset('gripper_action', data=gripper_actions)       # Action from /commands

            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['num_episodes'] = 1
            meta.attrs['fps'] = self.config['camera']['fps']
            meta.attrs['num_frames'] = n_frames

            if camera_info:
                meta.attrs['image_width'] = camera_info['width']
                meta.attrs['image_height'] = camera_info['height']
                meta.create_dataset('camera_K', data=np.array(camera_info['K']))
                if camera_info['D']:
                    meta.create_dataset('camera_D', data=np.array(camera_info['D']))

        print(f"Saved {n_frames} frames to {output_path}")

    def process(self, input_bag: str, output_dir: str) -> str:
        """Main processing pipeline."""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Extract data from bag
        print("\n=== Step 1: Extracting data from bag ===")
        extracted_data = self.extract_data_from_bag(input_bag, output_dir)

        # Step 2: Run offline SLAM
        print("\n=== Step 2: Running offline SLAM ===")
        rgb_dir = os.path.join(output_dir, 'rgb')
        depth_dir = os.path.join(output_dir, 'depth')
        trajectory = self.run_slam_offline(
            rgb_dir, depth_dir, output_dir,
            rgb_data=extracted_data.get('rgb', []),
            bag_path=input_bag
        )

        # Step 3: Synchronize data
        print("\n=== Step 3: Synchronizing data ===")
        synced_frames = self.synchronize_data(extracted_data, trajectory)

        # Step 4: Save to HDF5
        print("\n=== Step 4: Saving to HDF5 ===")
        hdf5_path = os.path.join(output_dir, 'dataset.hdf5')
        self.save_to_hdf5(synced_frames, extracted_data['camera_info'], hdf5_path)

        # Save metadata
        metadata = {
            'input_bag': input_bag,
            'num_frames': len(synced_frames),
            'config': self.config
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n=== Processing complete ===")
        print(f"Output directory: {output_dir}")
        print(f"HDF5 file: {hdf5_path}")

        return hdf5_path


def main():
    parser = argparse.ArgumentParser(description='Process ROS2 bag with SLAM')
    parser.add_argument('--input', '-i', required=True,
                        help='Input ROS2 bag directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for processed data')
    parser.add_argument('--config', '-c', default=None,
                        help='Path to config yaml file')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input bag not found: {args.input}")
        sys.exit(1)

    processor = BagProcessor(config_path=args.config)
    processor.process(args.input, args.output)


if __name__ == '__main__':
    main()
