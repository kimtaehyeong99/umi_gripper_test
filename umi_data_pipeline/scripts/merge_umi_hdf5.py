#!/usr/bin/env python3
"""
Merge multiple UMI format HDF5 files into one.

UMI format structure:
  data/
    camera0_rgb, robot0_eef_pos, robot0_eef_rot_axis_angle,
    robot0_gripper_width, robot0_demo_start_pose, robot0_demo_end_pose
  meta/
    episode_ends

Usage:
    # Merge specific files
    python3 merge_umi_hdf5.py -i ep1/dataset.hdf5 ep2/dataset.hdf5 ep3/dataset.hdf5 -o merged.hdf5

    # Merge all in directory
    python3 merge_umi_hdf5.py -d data/processed -o data/merged/dataset.hdf5
"""

import argparse
import os
import sys
from glob import glob
import h5py
import numpy as np


def merge_umi_hdf5_files(input_files: list, output_file: str):
    """
    Merge multiple UMI format HDF5 files into one.

    - Concatenates all data arrays
    - Updates episode_ends with cumulative indices
    """
    print(f"Merging {len(input_files)} UMI HDF5 files...")

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Collect all data
    all_data = {}
    all_episode_ends = []
    total_frames = 0

    for input_file in input_files:
        print(f"\nProcessing: {input_file}")

        with h5py.File(input_file, 'r') as f:
            # Check if UMI format
            if 'data' not in f:
                print(f"  Warning: {input_file} is not UMI format (no 'data' group), skipping")
                continue

            data_group = f['data']
            meta_group = f['meta'] if 'meta' in f else None

            # Get number of frames in this file
            first_key = list(data_group.keys())[0]
            n_frames = data_group[first_key].shape[0]
            print(f"  Frames: {n_frames}")

            # Collect data arrays
            for key in data_group.keys():
                arr = data_group[key][:]
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(arr)
                print(f"    {key}: {arr.shape}")

            # Update episode ends
            if meta_group and 'episode_ends' in meta_group:
                episode_ends = meta_group['episode_ends'][:]
                # Adjust indices for concatenation
                adjusted_ends = episode_ends + total_frames
                all_episode_ends.extend(adjusted_ends.tolist())
            else:
                # Single episode, ends at total_frames + n_frames
                all_episode_ends.append(total_frames + n_frames)

            total_frames += n_frames

    if total_frames == 0:
        print("Error: No frames found in any input file")
        return 0

    # Write merged file
    print(f"\nWriting merged file: {output_file}")
    print(f"Total frames: {total_frames}")
    print(f"Total episodes: {len(all_episode_ends)}")

    with h5py.File(output_file, 'w') as f:
        # Create data group
        data_group = f.create_group('data')

        for key, arrays in all_data.items():
            merged = np.concatenate(arrays, axis=0)
            data_group.create_dataset(key, data=merged, compression='lzf')
            print(f"  {key}: {merged.shape}")

        # Create meta group
        meta_group = f.create_group('meta')
        episode_ends = np.array(all_episode_ends, dtype=np.int64)
        meta_group.create_dataset('episode_ends', data=episode_ends)
        print(f"  episode_ends: {episode_ends}")

        # Attributes
        f.attrs['total_frames'] = total_frames
        f.attrs['num_episodes'] = len(all_episode_ends)

    print(f"\nMerged {len(input_files)} files ({len(all_episode_ends)} episodes, {total_frames} frames)")
    return total_frames


def main():
    parser = argparse.ArgumentParser(description='Merge multiple UMI format HDF5 files')
    parser.add_argument('--inputs', '-i', nargs='+',
                        help='Input HDF5 files')
    parser.add_argument('--input-dir', '-d',
                        help='Directory containing processed folders (each with dataset.hdf5)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output merged HDF5 file')

    args = parser.parse_args()

    # Collect input files
    input_files = []

    if args.inputs:
        input_files = args.inputs
    elif args.input_dir:
        # Find all dataset.hdf5 files in subdirectories
        pattern = os.path.join(args.input_dir, '*', 'dataset.hdf5')
        input_files = sorted(glob(pattern))

        if not input_files:
            # Try direct hdf5 files
            pattern = os.path.join(args.input_dir, '*.hdf5')
            input_files = sorted(glob(pattern))

    if not input_files:
        print("Error: No input files found")
        print("Usage:")
        print("  python3 merge_umi_hdf5.py -i file1.hdf5 file2.hdf5 -o merged.hdf5")
        print("  python3 merge_umi_hdf5.py -d data/processed -o merged.hdf5")
        sys.exit(1)

    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f}")

    # Verify all files exist
    for f in input_files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)

    merge_umi_hdf5_files(input_files, args.output)


if __name__ == '__main__':
    main()
