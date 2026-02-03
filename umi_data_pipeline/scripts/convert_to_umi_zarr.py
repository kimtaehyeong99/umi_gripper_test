#!/usr/bin/env python3
"""
UMI Data Pipeline - Convert HDF5 to UMI Zarr Format

This script converts the processed HDF5 data to UMI-compatible Zarr format:
- robot0_eef_pos: [3] End-effector position
- robot0_eef_rot_axis_angle: [6] 6D rotation representation
- robot0_gripper_width: [1] Gripper width
- camera0_rgb: [3, 224, 224] RGB images
- action: [10] = 3 pos + 6 rot6d + 1 gripper
- episode_ends: Episode boundary indices

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
    from zarr import blosc
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    print("Warning: zarr not installed. Install with: pip install zarr")

try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Install with: pip install scipy")


def quaternion_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 6D rotation representation.

    The 6D representation uses the first two columns of the rotation matrix,
    which is a continuous representation for rotations.

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

    Args:
        img: RGB image [H, W, 3]
        target_size: Target size (width, height)

    Returns:
        Preprocessed image [3, H, W] uint8
    """
    # Resize
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # HWC -> CHW
    img_chw = np.transpose(img_resized, (2, 0, 1))

    return img_chw.astype(np.uint8)


def compute_actions(positions: np.ndarray, rot_6ds: np.ndarray,
                    gripper_widths: np.ndarray) -> np.ndarray:
    """
    Compute actions from states.

    In UMI, actions are the state at time t+1 (absolute actions).
    Action: [pos(3), rot6d(6), gripper(1)] = 10 dims

    Args:
        positions: [N, 3] End-effector positions
        rot_6ds: [N, 6] 6D rotations
        gripper_widths: [N, 1] Gripper widths

    Returns:
        actions: [N, 10] Actions
    """
    n_frames = len(positions)

    # Action is next state (shifted by 1)
    # For last frame, repeat the last action
    actions = np.zeros((n_frames, 10), dtype=np.float32)

    for i in range(n_frames):
        next_idx = min(i + 1, n_frames - 1)
        actions[i, :3] = positions[next_idx]
        actions[i, 3:9] = rot_6ds[next_idx]
        actions[i, 9] = gripper_widths[next_idx]

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
            all_rot_6ds = []
            all_gripper_widths = []
            all_images = []
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

                # Convert quaternions to 6D rotation
                rot_6ds = np.array([quaternion_to_rotation_6d(q) for q in quaternions])

                # Gripper widths
                gripper_widths = episode['gripper_width'][:].reshape(-1, 1)

                # Images - preprocess to target size
                rgb_images = episode['rgb_images'][:]
                processed_images = np.array([
                    preprocess_image(img, self.image_size)
                    for img in rgb_images
                ])

                # Accumulate
                all_positions.append(positions)
                all_rot_6ds.append(rot_6ds)
                all_gripper_widths.append(gripper_widths)
                all_images.append(processed_images)

                total_frames += n_frames
                episode_ends.append(total_frames)

            # Stack all episodes
            all_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
            all_rot_6ds = np.concatenate(all_rot_6ds, axis=0).astype(np.float32)
            all_gripper_widths = np.concatenate(all_gripper_widths, axis=0).astype(np.float32)
            all_images = np.concatenate(all_images, axis=0)
            episode_ends = np.array(episode_ends, dtype=np.int64)

            # Compute actions
            all_actions = compute_actions(all_positions, all_rot_6ds, all_gripper_widths)

            print(f"\nTotal frames: {total_frames}")
            print(f"Positions shape: {all_positions.shape}")
            print(f"Rotations shape: {all_rot_6ds.shape}")
            print(f"Gripper shape: {all_gripper_widths.shape}")
            print(f"Images shape: {all_images.shape}")
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
        compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.SHUFFLE)

        print("\nSaving to Zarr...")

        # robot0_eef_pos
        data.create_dataset(
            'robot0_eef_pos',
            data=all_positions,
            chunks=(1000, 3),
            compressor=compressor
        )
        print("  - robot0_eef_pos")

        # robot0_eef_rot_axis_angle (using 6D representation)
        data.create_dataset(
            'robot0_eef_rot_axis_angle',
            data=all_rot_6ds,
            chunks=(1000, 6),
            compressor=compressor
        )
        print("  - robot0_eef_rot_axis_angle")

        # robot0_gripper_width
        data.create_dataset(
            'robot0_gripper_width',
            data=all_gripper_widths,
            chunks=(1000, 1),
            compressor=compressor
        )
        print("  - robot0_gripper_width")

        # camera0_rgb (large, use chunking)
        data.create_dataset(
            'camera0_rgb',
            data=all_images,
            chunks=(100, 3, self.image_size[1], self.image_size[0]),
            compressor=compressor
        )
        print("  - camera0_rgb")

        # action
        data.create_dataset(
            'action',
            data=all_actions,
            chunks=(1000, 10),
            compressor=compressor
        )
        print("  - action")

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
