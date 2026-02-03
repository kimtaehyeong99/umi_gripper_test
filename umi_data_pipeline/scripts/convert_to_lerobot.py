#!/usr/bin/env python3
"""
UMI Data Pipeline - Convert HDF5 to LeRobot Format

This script converts the processed HDF5 data to LeRobot-compatible format.
LeRobot format documentation: https://github.com/huggingface/lerobot

Usage:
    python3 convert_to_lerobot.py --input data/processed/session/dataset.hdf5 --output data/datasets/lerobot_demo/
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import h5py

# LeRobot format is still evolving, this is a basic implementation
# that can be extended as the LeRobot format stabilizes


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 to LeRobot format')
    parser.add_argument('--input', '-i', required=True,
                        help='Input HDF5 file path')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for LeRobot dataset')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (width height)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {args.input} to LeRobot format at {output_dir}")

    # Read HDF5
    with h5py.File(args.input, 'r') as f:
        episodes = [k for k in f.keys() if k.startswith('episode_')]
        print(f"Found {len(episodes)} episodes")

        # Create LeRobot directory structure
        images_dir = output_dir / 'data' / 'observation.images.camera0'
        images_dir.mkdir(parents=True, exist_ok=True)

        all_data = {
            'observation.state': [],
            'action': [],
            'episode_index': [],
            'frame_index': [],
            'timestamp': [],
        }

        global_frame_idx = 0

        for ep_idx, episode_name in enumerate(sorted(episodes)):
            episode = f[episode_name]
            n_frames = episode['rgb_images'].shape[0]
            print(f"  Processing {episode_name}: {n_frames} frames")

            # Extract data
            poses = episode['camera_pose'][:]
            gripper = episode['gripper_width'][:]
            timestamps = episode['timestamps'][:] if 'timestamps' in episode else np.arange(n_frames) / 30.0
            rgb_images = episode['rgb_images'][:]

            for i in range(n_frames):
                # Save image
                img = rgb_images[i]
                img = cv2.resize(img, tuple(args.image_size))
                img_path = images_dir / f'{global_frame_idx:06d}.png'
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # State: [pos(3), quat(4), gripper(1)] = 8 dims
                state = np.concatenate([poses[i], [gripper[i].item() if gripper[i].ndim == 0 else gripper[i][0]]])
                all_data['observation.state'].append(state)

                # Action: next state (or same for last frame)
                next_i = min(i + 1, n_frames - 1)
                action = np.concatenate([poses[next_i], [gripper[next_i].item() if gripper[next_i].ndim == 0 else gripper[next_i][0]]])
                all_data['action'].append(action)

                all_data['episode_index'].append(ep_idx)
                all_data['frame_index'].append(i)
                all_data['timestamp'].append(timestamps[i])

                global_frame_idx += 1

    # Convert to numpy arrays
    for key in all_data:
        all_data[key] = np.array(all_data[key])

    # Save as numpy files
    np.save(output_dir / 'data' / 'observation.state.npy', all_data['observation.state'])
    np.save(output_dir / 'data' / 'action.npy', all_data['action'])
    np.save(output_dir / 'data' / 'episode_index.npy', all_data['episode_index'])
    np.save(output_dir / 'data' / 'frame_index.npy', all_data['frame_index'])
    np.save(output_dir / 'data' / 'timestamp.npy', all_data['timestamp'])

    # Create metadata
    metadata = {
        'robot_type': 'umi_gripper',
        'fps': 30,
        'num_episodes': len(episodes),
        'total_frames': global_frame_idx,
        'image_size': args.image_size,
        'observation_space': {
            'state': {'shape': [8], 'dtype': 'float32'},
            'images.camera0': {'shape': [3] + args.image_size[::-1], 'dtype': 'uint8'},
        },
        'action_space': {
            'shape': [8],
            'dtype': 'float32'
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Total frames: {global_frame_idx}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
