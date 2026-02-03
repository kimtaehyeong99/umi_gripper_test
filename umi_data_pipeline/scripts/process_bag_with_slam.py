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


class BagProcessor:
    """Process ROS2 bag files for UMI dataset creation."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from yaml file."""
        default_config = {
            'camera': {
                'rgb_topic': '/camera/camera/color/image_rect_raw',
                'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/camera/camera/color/camera_info',
                'fps': 30,
            },
            'gripper': {
                'topic': '/trigger_position_controller/commands',
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
        gripper_topic = self.config['gripper']['topic']

        rgb_data = []
        depth_data = []
        gripper_data = []
        camera_info = None

        with Reader(bag_path) as reader:
            # Get connections for our topics
            connections = {}
            for conn in reader.connections:
                if conn.topic in [rgb_topic, depth_topic, gripper_topic,
                                  self.config['camera']['camera_info_topic']]:
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

                elif topic == gripper_topic:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    width = self._decode_gripper(msg)
                    gripper_data.append({
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

        print(f"Extracted: {len(rgb_data)} RGB, {len(depth_data)} Depth, {len(gripper_data)} Gripper frames")

        return {
            'rgb': rgb_data,
            'depth': depth_data,
            'gripper': gripper_data,
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

    def _decode_gripper(self, msg) -> float:
        """Decode gripper message to width in meters."""
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
            print(f"Error decoding gripper: {e}")
            return 0.0

    def run_slam_offline(self, rgb_dir: str, depth_dir: str, output_dir: str) -> List[Dict]:
        """
        Run ORB-SLAM3 offline on extracted images.

        This uses ros2 bag play + SLAM node approach.
        Alternative: Direct image processing with ORB-SLAM3 library.
        """
        print("Running offline SLAM...")

        # For now, create a placeholder trajectory
        # In production, this would run ORB-SLAM3 on the images

        rgb_files = sorted(Path(rgb_dir).glob('*.png'))
        trajectory = []

        # Placeholder: identity poses (to be replaced with actual SLAM)
        for i, rgb_path in enumerate(rgb_files):
            trajectory.append({
                'frame_idx': i,
                'timestamp': i / 30.0,  # Assuming 30 fps
                'position': [0.0, 0.0, 0.0],
                'quaternion': [0.0, 0.0, 0.0, 1.0],  # x, y, z, w
                'is_valid': True
            })

        print(f"Generated {len(trajectory)} pose estimates")
        print("Note: Replace with actual ORB-SLAM3 processing for real poses")

        # Save trajectory to CSV
        traj_path = os.path.join(output_dir, 'camera_trajectory.csv')
        with open(traj_path, 'w') as f:
            f.write('frame_idx,timestamp,x,y,z,qx,qy,qz,qw,is_valid\n')
            for pose in trajectory:
                f.write(f"{pose['frame_idx']},{pose['timestamp']:.6f},"
                       f"{pose['position'][0]:.6f},{pose['position'][1]:.6f},{pose['position'][2]:.6f},"
                       f"{pose['quaternion'][0]:.6f},{pose['quaternion'][1]:.6f},"
                       f"{pose['quaternion'][2]:.6f},{pose['quaternion'][3]:.6f},"
                       f"{1 if pose['is_valid'] else 0}\n")

        return trajectory

    def synchronize_data(self, extracted_data: Dict, trajectory: List[Dict]) -> List[Dict]:
        """Synchronize RGB, Depth, Gripper, and Pose data by timestamp."""

        print("Synchronizing data...")

        rgb_data = extracted_data['rgb']
        depth_data = extracted_data['depth']
        gripper_data = extracted_data['gripper']

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

            # Find closest gripper
            gripper_idx = self._find_closest_timestamp(
                rgb_ts, [g['timestamp'] for g in gripper_data]
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
                'gripper_width': gripper_data[gripper_idx]['width'] if gripper_idx is not None else 0.0,
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
            gripper_widths = np.array([frame['gripper_width'] for frame in synced_frames])

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
            episode.create_dataset('gripper_width', data=gripper_widths)

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
        trajectory = self.run_slam_offline(rgb_dir, depth_dir, output_dir)

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
