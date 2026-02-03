#!/usr/bin/env python3
"""
UMI Data Pipeline - Convert HDF5 to UMI Zarr Format

This script converts the processed HDF5 data to UMI-compatible Zarr format:
- robot0_eef_pos: [N, 3] End-effector position (meters)
- robot0_eef_rot_axis_angle: [N, 3] Rotation as axis-angle (rotvec)
- robot0_gripper_width: [N, 1] Gripper width (meters)
- robot0_demo_start_pose: [N, 6] Episode start pose (pos + axis_angle)
- robot0_demo_end_pose: [N, 6] Episode end pose (pos + axis_angle)
- camera0_rgb: [N, H, W, 3] RGB images (HWC format)
- action: [N, 10] = pos(3) + rot6d(6) + gripper(1)
- meta/episode_ends: Episode boundary indices

Usage:
    python3 convert_to_umi_zarr.py --input data/processed/session/dataset.hdf5 --output data/datasets/umi_demo.zarr.zip
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import h5py

try:
    import zarr
    from numcodecs import Blosc
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    print("Warning: zarr not installed. Install with: pip install 'zarr<3' numcodecs")

try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Install with: pip install scipy")


def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle (rotation vector) representation.

    UMI stores rotations as axis-angle [3] in the Zarr file.

    Args:
        quat: Quaternion [x, y, z, w]

    Returns:
        Axis-angle [3] (rotation vector where magnitude is angle in radians)
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(3, dtype=np.float32)

    # scipy uses [x, y, z, w] format
    rotvec = Rotation.from_quat(quat).as_rotvec()
    return rotvec.astype(np.float32)


def quaternion_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 6D rotation representation.

    The 6D representation uses the first two columns of the rotation matrix,
    which is a continuous representation for rotations.
    Used for action representation.

    Args:
        quat: Quaternion [x, y, z, w]

    Returns:
        6D rotation [r11, r21, r31, r12, r22, r32]
    """
    if not SCIPY_AVAILABLE:
        # Fallback: identity rotation if scipy not available
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # scipy uses [x, y, z, w] format
    R = Rotation.from_quat(quat).as_matrix()

    # 6D representation: first two columns of rotation matrix, flattened column-wise
    rot_6d = np.concatenate([R[:, 0], R[:, 1]])

    return rot_6d.astype(np.float32)


def axis_angle_to_rotation_6d(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle to 6D rotation representation.

    Used for computing actions from stored axis-angle observations.

    Args:
        rotvec: Axis-angle [3]

    Returns:
        6D rotation [6]
    """
    if not SCIPY_AVAILABLE:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    R = Rotation.from_rotvec(rotvec).as_matrix()
    rot_6d = np.concatenate([R[:, 0], R[:, 1]])
    return rot_6d.astype(np.float32)


def rotation_6d_to_axis_angle(rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation to axis-angle representation.

    UMI uses axis-angle internally but stores as 6D for continuity.
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(3, dtype=np.float32)

    # Reconstruct rotation matrix from 6D
    a1 = rot_6d[:3]
    a2 = rot_6d[3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)

    R = np.column_stack([b1, b2, b3])

    # Convert to axis-angle
    rotvec = Rotation.from_matrix(R).as_rotvec()

    return rotvec.astype(np.float32)


def preprocess_image(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for UMI format.

    UMI uses HWC format (Height, Width, Channels).

    Args:
        img: RGB image [H, W, 3]
        target_size: Target size (width, height)

    Returns:
        Preprocessed image [H, W, 3] uint8 (HWC format)
    """
    # Resize to target size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Keep HWC format (UMI standard)
    return img_resized.astype(np.uint8)


def compute_actions(positions: np.ndarray, axis_angles: np.ndarray,
                    gripper_widths: np.ndarray) -> np.ndarray:
    """
    Compute actions from states.

    In UMI, actions are the state at time t+1 (absolute actions).
    Action: [pos(3), rot6d(6), gripper(1)] = 10 dims

    Note: Observations store axis-angle [3], but actions use rot6d [6].

    Args:
        positions: [N, 3] End-effector positions
        axis_angles: [N, 3] Axis-angle rotations
        gripper_widths: [N, 1] Gripper widths

    Returns:
        actions: [N, 10] Actions with rot6d representation
    """
    n_frames = len(positions)

    # Convert axis-angles to rot6d for actions
    rot_6ds = np.array([axis_angle_to_rotation_6d(aa) for aa in axis_angles])

    # Action is next state (shifted by 1)
    # For last frame, repeat the last action
    actions = np.zeros((n_frames, 10), dtype=np.float32)

    for i in range(n_frames):
        next_idx = min(i + 1, n_frames - 1)
        actions[i, :3] = positions[next_idx]
        actions[i, 3:9] = rot_6ds[next_idx]
        actions[i, 9] = gripper_widths[next_idx].item() if gripper_widths[next_idx].ndim > 0 else gripper_widths[next_idx]

    return actions


class UMIZarrConverter:
    """Convert HDF5 dataset to UMI Zarr format."""

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size

    def convert(self, input_hdf5: str, output_zarr: str) -> None:
        """
        Convert HDF5 to UMI Zarr format.

        Args:
            input_hdf5: Path to input HDF5 file
            output_zarr: Path to output Zarr file (can be .zarr or .zarr.zip)
        """
        if not ZARR_AVAILABLE:
            raise ImportError("zarr package required. Install with: pip install zarr")

        print(f"Converting {input_hdf5} to {output_zarr}")

        # Open input HDF5
        with h5py.File(input_hdf5, 'r') as hdf5_file:
            # Find all episodes
            episodes = [k for k in hdf5_file.keys() if k.startswith('episode_')]
            print(f"Found {len(episodes)} episodes")

            if len(episodes) == 0:
                print("Error: No episodes found in HDF5 file")
                return

            # Collect all data
            all_positions = []
            all_axis_angles = []
            all_gripper_observations = []  # Observation from /joint_states
            all_gripper_actions = []       # Action from /gripper_position_controller/commands
            all_images = []
            all_demo_start_poses = []
            all_demo_end_poses = []
            episode_ends = []

            total_frames = 0

            for episode_name in sorted(episodes):
                episode = hdf5_file[episode_name]
                n_frames = episode['rgb_images'].shape[0]
                print(f"  {episode_name}: {n_frames} frames")

                # Camera poses [N, 7] -> position [N, 3] and quaternion [N, 4]
                poses = episode['camera_pose'][:]
                positions = poses[:, :3]
                quaternions = poses[:, 3:7]

                # Convert quaternions to axis-angle (UMI standard storage)
                axis_angles = np.array([quaternion_to_axis_angle(q) for q in quaternions])

                # Gripper observation (from /joint_states) - used for state observation
                gripper_observations = episode['gripper_width'][:].reshape(-1, 1)

                # Gripper action (from /commands) - used for action computation
                if 'gripper_action' in episode:
                    gripper_actions = episode['gripper_action'][:].reshape(-1, 1)
                else:
                    # Fallback: use observation if action not available
                    gripper_actions = gripper_observations.copy()

                # Images - preprocess to target size (HWC format)
                rgb_images = episode['rgb_images'][:]
                processed_images = np.array([
                    preprocess_image(img, self.image_size)
                    for img in rgb_images
                ])

                # Demo start/end poses [6] = pos(3) + axis_angle(3)
                # Broadcast to all frames in episode
                demo_start_pose = np.concatenate([positions[0], axis_angles[0]])
                demo_end_pose = np.concatenate([positions[-1], axis_angles[-1]])
                demo_start_poses = np.tile(demo_start_pose, (n_frames, 1))
                demo_end_poses = np.tile(demo_end_pose, (n_frames, 1))

                # Accumulate
                all_positions.append(positions)
                all_axis_angles.append(axis_angles)
                all_gripper_observations.append(gripper_observations)
                all_gripper_actions.append(gripper_actions)
                all_images.append(processed_images)
                all_demo_start_poses.append(demo_start_poses)
                all_demo_end_poses.append(demo_end_poses)

                total_frames += n_frames
                episode_ends.append(total_frames)

            # Stack all episodes
            all_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
            all_axis_angles = np.concatenate(all_axis_angles, axis=0).astype(np.float32)
            all_gripper_observations = np.concatenate(all_gripper_observations, axis=0).astype(np.float32)
            all_gripper_actions = np.concatenate(all_gripper_actions, axis=0).astype(np.float32)
            all_images = np.concatenate(all_images, axis=0)
            all_demo_start_poses = np.concatenate(all_demo_start_poses, axis=0).astype(np.float32)
            all_demo_end_poses = np.concatenate(all_demo_end_poses, axis=0).astype(np.float32)
            episode_ends = np.array(episode_ends, dtype=np.int64)

            # Compute actions using gripper_action (from commands), not gripper_observation
            all_actions = compute_actions(all_positions, all_axis_angles, all_gripper_actions)

            print(f"\nTotal frames: {total_frames}")
            print(f"Positions shape: {all_positions.shape}")
            print(f"Axis-angles shape: {all_axis_angles.shape}")
            print(f"Gripper observation shape: {all_gripper_observations.shape}")
            print(f"Gripper action shape: {all_gripper_actions.shape}")
            print(f"Images shape: {all_images.shape}")
            print(f"Demo start poses shape: {all_demo_start_poses.shape}")
            print(f"Demo end poses shape: {all_demo_end_poses.shape}")
            print(f"Actions shape: {all_actions.shape}")
            print(f"Episode ends: {episode_ends}")

        # Create Zarr store
        if output_zarr.endswith('.zip'):
            store = zarr.ZipStore(output_zarr, mode='w')
        else:
            store = zarr.DirectoryStore(output_zarr)

        root = zarr.group(store=store)

        # Create data group
        data = root.create_group('data')

        # Store arrays with compression
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

        print("\nSaving to Zarr...")

        # robot0_eef_pos [N, 3]
        data.create_dataset(
            'robot0_eef_pos',
            data=all_positions,
            chunks=(1000, 3),
            compressor=compressor
        )
        print("  - robot0_eef_pos [N, 3]")

        # robot0_eef_rot_axis_angle [N, 3] - UMI stores as axis-angle
        data.create_dataset(
            'robot0_eef_rot_axis_angle',
            data=all_axis_angles,
            chunks=(1000, 3),
            compressor=compressor
        )
        print("  - robot0_eef_rot_axis_angle [N, 3]")

        # robot0_gripper_width [N, 1] - Observation from /joint_states
        data.create_dataset(
            'robot0_gripper_width',
            data=all_gripper_observations,
            chunks=(1000, 1),
            compressor=compressor
        )
        print("  - robot0_gripper_width [N, 1] (observation)")

        # robot0_demo_start_pose [N, 6] = pos(3) + axis_angle(3)
        data.create_dataset(
            'robot0_demo_start_pose',
            data=all_demo_start_poses,
            chunks=(1000, 6),
            compressor=compressor
        )
        print("  - robot0_demo_start_pose [N, 6]")

        # robot0_demo_end_pose [N, 6] = pos(3) + axis_angle(3)
        data.create_dataset(
            'robot0_demo_end_pose',
            data=all_demo_end_poses,
            chunks=(1000, 6),
            compressor=compressor
        )
        print("  - robot0_demo_end_pose [N, 6]")

        # camera0_rgb [N, H, W, 3] - HWC format (UMI standard)
        data.create_dataset(
            'camera0_rgb',
            data=all_images,
            chunks=(1, self.image_size[1], self.image_size[0], 3),
            compressor=compressor
        )
        print(f"  - camera0_rgb [N, {self.image_size[1]}, {self.image_size[0]}, 3]")

        # action [N, 10] = pos(3) + rot6d(6) + gripper(1)
        data.create_dataset(
            'action',
            data=all_actions,
            chunks=(1000, 10),
            compressor=compressor
        )
        print("  - action [N, 10]")

        # Create meta group
        meta = root.create_group('meta')
        meta.create_dataset('episode_ends', data=episode_ends)
        print("  - episode_ends")

        # Add attributes
        root.attrs['total_frames'] = total_frames
        root.attrs['num_episodes'] = len(episode_ends)
        root.attrs['image_size'] = list(self.image_size)

        # Close store
        if hasattr(store, 'close'):
            store.close()

        print(f"\nConversion complete: {output_zarr}")


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 to UMI Zarr format')
    parser.add_argument('--input', '-i', required=True,
                        help='Input HDF5 file path')
    parser.add_argument('--output', '-o', required=True,
                        help='Output Zarr path (.zarr or .zarr.zip)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (width height)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    converter = UMIZarrConverter(image_size=tuple(args.image_size))
    converter.convert(args.input, args.output)


if __name__ == '__main__':
    main()
