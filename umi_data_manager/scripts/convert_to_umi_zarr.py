#!/usr/bin/env python3
"""
UMI Data Pipeline - Convert HDF5 to UMI Zarr Format

This script converts the processed HDF5 data to UMI-compatible Zarr format:
- robot0_eef_pos: [N, 3] End-effector position (meters)
- robot0_eef_rot_axis_angle: [N, 3] Rotation as axis-angle (rotvec)
- robot0_gripper_width: [N, 1] Gripper width (meters)
- camera0_rgb: [N, H, W, 3] RGB images (HWC format)
- meta/episode_ends: Episode boundary indices

Note: action, demo_start_pose, demo_end_pose are computed during training
      by the UMI training pipeline (sampler.py), not stored in Zarr.

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

# JPEG XL codec for image compression (same as official UMI)
try:
    from imagecodecs_numcodecs import JpegXl, register_codecs
    register_codecs(verbose=False)
    JPEGXL_AVAILABLE = True
except ImportError:
    JPEGXL_AVAILABLE = False
    print("Warning: imagecodecs not available. Using Blosc for images instead of JPEG XL.")

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


class UMIZarrConverter:
    """Convert HDF5 dataset to UMI Zarr format."""

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size

    def convert(self, input_hdf5: str, output_zarr: str) -> None:
        """
        Convert HDF5 to UMI Zarr format.

        Supports both:
        - New UMI format (data/ group with pre-converted axis-angle)
        - Legacy format (episode_XXXX/ groups with quaternions)

        Args:
            input_hdf5: Path to input HDF5 file
            output_zarr: Path to output Zarr file (can be .zarr or .zarr.zip)
        """
        if not ZARR_AVAILABLE:
            raise ImportError("zarr package required. Install with: pip install zarr")

        print(f"Converting {input_hdf5} to {output_zarr}")

        # Open input HDF5
        with h5py.File(input_hdf5, 'r') as hdf5_file:
            # Detect format: new UMI format (data/ group) or legacy (episode_XXXX/)
            is_new_format = 'data' in hdf5_file

            if is_new_format:
                # New UMI format - data is already in correct format
                print("Detected new UMI HDF5 format")
                all_positions, all_axis_angles, all_gripper_observations, \
                all_images, all_demo_start_poses, all_demo_end_poses, \
                episode_ends, total_frames = self._load_new_format(hdf5_file)
            else:
                # Legacy format - need conversion
                print("Detected legacy HDF5 format")
                all_positions, all_axis_angles, all_gripper_observations, \
                all_images, all_demo_start_poses, all_demo_end_poses, \
                episode_ends, total_frames = self._load_legacy_format(hdf5_file)

            print(f"\nTotal frames: {total_frames}")
            print(f"Positions shape: {all_positions.shape}")
            print(f"Axis-angles shape: {all_axis_angles.shape}")
            print(f"Gripper observation shape: {all_gripper_observations.shape}")
            print(f"Images shape: {all_images.shape}")
            print(f"Demo start poses shape: {all_demo_start_poses.shape}")
            print(f"Demo end poses shape: {all_demo_end_poses.shape}")
            print(f"Episode ends: {episode_ends}")

        # Save to Zarr
        self._save_to_zarr(output_zarr, all_positions, all_axis_angles,
                          all_gripper_observations, all_images,
                          all_demo_start_poses, all_demo_end_poses,
                          episode_ends, total_frames)

    def _load_new_format(self, hdf5_file):
        """Load data from new UMI HDF5 format (data/ group)."""
        data = hdf5_file['data']
        meta = hdf5_file['meta']

        # Load pre-converted data
        all_positions = data['robot0_eef_pos'][:].astype(np.float32)
        all_axis_angles = data['robot0_eef_rot_axis_angle'][:].astype(np.float32)
        all_gripper_observations = data['robot0_gripper_width'][:].astype(np.float32)
        all_images = data['camera0_rgb'][:]  # Already 224x224

        # Load demo_start_pose and demo_end_pose if available, otherwise compute
        if 'robot0_demo_start_pose' in data:
            all_demo_start_poses = data['robot0_demo_start_pose'][:].astype(np.float32)
            all_demo_end_poses = data['robot0_demo_end_pose'][:].astype(np.float32)
        else:
            # Compute from episode boundaries
            episode_ends = meta['episode_ends'][:].astype(np.int64)
            total_frames = len(all_positions)
            all_demo_start_poses = np.zeros((total_frames, 6), dtype=np.float32)
            all_demo_end_poses = np.zeros((total_frames, 6), dtype=np.float32)

            start_idx = 0
            for end_idx in episode_ends:
                # Start pose for this episode
                start_pose = np.concatenate([all_positions[start_idx], all_axis_angles[start_idx]])
                end_pose = np.concatenate([all_positions[end_idx-1], all_axis_angles[end_idx-1]])
                all_demo_start_poses[start_idx:end_idx] = start_pose
                all_demo_end_poses[start_idx:end_idx] = end_pose
                start_idx = end_idx

        episode_ends = meta['episode_ends'][:].astype(np.int64)
        total_frames = len(all_positions)

        return (all_positions, all_axis_angles, all_gripper_observations,
                all_images, all_demo_start_poses, all_demo_end_poses,
                episode_ends, total_frames)

    def _load_legacy_format(self, hdf5_file):
        """Load data from legacy HDF5 format (episode_XXXX/ groups)."""
        episodes = [k for k in hdf5_file.keys() if k.startswith('episode_')]
        print(f"Found {len(episodes)} episodes")

        if len(episodes) == 0:
            raise ValueError("No episodes found in HDF5 file")

        # Collect all data
        all_positions = []
        all_axis_angles = []
        all_gripper_observations = []
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

            # Gripper observation (from /joint_states)
            gripper_observations = episode['gripper_width'][:].reshape(-1, 1)

            # Images - preprocess to target size (HWC format)
            rgb_images = episode['rgb_images'][:]
            processed_images = np.array([
                preprocess_image(img, self.image_size)
                for img in rgb_images
            ])

            # Compute demo_start_pose and demo_end_pose for this episode
            start_pose = np.concatenate([positions[0], axis_angles[0]])
            end_pose = np.concatenate([positions[-1], axis_angles[-1]])
            demo_start_poses = np.tile(start_pose, (n_frames, 1))
            demo_end_poses = np.tile(end_pose, (n_frames, 1))

            # Accumulate
            all_positions.append(positions)
            all_axis_angles.append(axis_angles)
            all_gripper_observations.append(gripper_observations)
            all_images.append(processed_images)
            all_demo_start_poses.append(demo_start_poses)
            all_demo_end_poses.append(demo_end_poses)

            total_frames += n_frames
            episode_ends.append(total_frames)

        # Stack all episodes
        all_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
        all_axis_angles = np.concatenate(all_axis_angles, axis=0).astype(np.float32)
        all_gripper_observations = np.concatenate(all_gripper_observations, axis=0).astype(np.float32)
        all_images = np.concatenate(all_images, axis=0)
        all_demo_start_poses = np.concatenate(all_demo_start_poses, axis=0).astype(np.float32)
        all_demo_end_poses = np.concatenate(all_demo_end_poses, axis=0).astype(np.float32)
        episode_ends = np.array(episode_ends, dtype=np.int64)

        return (all_positions, all_axis_angles, all_gripper_observations,
                all_images, all_demo_start_poses, all_demo_end_poses,
                episode_ends, total_frames)

    def _save_to_zarr(self, output_zarr, all_positions, all_axis_angles,
                      all_gripper_observations, all_images,
                      all_demo_start_poses, all_demo_end_poses,
                      episode_ends, total_frames):
        """Save all data to Zarr format."""
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
        print("  - robot0_gripper_width [N, 1]")

        # robot0_demo_start_pose [N, 6] - Episode start pose (required for training)
        data.create_dataset(
            'robot0_demo_start_pose',
            data=all_demo_start_poses,
            chunks=(1000, 6),
            compressor=compressor
        )
        print("  - robot0_demo_start_pose [N, 6]")

        # robot0_demo_end_pose [N, 6] - Episode end pose (required for training)
        data.create_dataset(
            'robot0_demo_end_pose',
            data=all_demo_end_poses,
            chunks=(1000, 6),
            compressor=compressor
        )
        print("  - robot0_demo_end_pose [N, 6]")

        # camera0_rgb [N, H, W, 3] - HWC format (UMI standard)
        # Use JPEG XL compression (same as official UMI) if available
        if JPEGXL_AVAILABLE:
            img_compressor = JpegXl(level=99, numthreads=1)
            print("  Using JPEG XL compression for images (same as official UMI)")
        else:
            img_compressor = compressor
            print("  Using Blosc compression for images (JPEG XL not available)")

        data.create_dataset(
            'camera0_rgb',
            data=all_images,
            chunks=(1, self.image_size[1], self.image_size[0], 3),
            compressor=img_compressor
        )
        print(f"  - camera0_rgb [N, {self.image_size[1]}, {self.image_size[0]}, 3]")

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
