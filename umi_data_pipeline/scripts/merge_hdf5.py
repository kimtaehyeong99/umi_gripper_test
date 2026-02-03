#!/usr/bin/env python3
"""
Merge multiple HDF5 files into one.

Usage:
    python3 merge_hdf5.py --inputs ep1/dataset.hdf5 ep2/dataset.hdf5 ep3/dataset.hdf5 --output merged.hdf5

    # Or with glob pattern:
    python3 merge_hdf5.py --input-dir data/processed --output data/merged/dataset.hdf5
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
import h5py
import numpy as np


def merge_hdf5_files(input_files: list, output_file: str):
    """
    Merge multiple HDF5 files into one.

    Each input file's episodes are renumbered sequentially.
    """
    print(f"Merging {len(input_files)} HDF5 files...")

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    episode_counter = 0

    with h5py.File(output_file, 'w') as out_f:
        for input_file in input_files:
            print(f"\nProcessing: {input_file}")

            with h5py.File(input_file, 'r') as in_f:
                # Get all episode groups
                episodes = [k for k in in_f.keys() if k.startswith('episode_')]
                episodes = sorted(episodes)

                print(f"  Found {len(episodes)} episodes")

                for old_name in episodes:
                    new_name = f"episode_{episode_counter:03d}"
                    print(f"    {old_name} -> {new_name}")

                    # Copy group with new name
                    in_f.copy(old_name, out_f, name=new_name)
                    episode_counter += 1

        # Copy attributes if any
        out_f.attrs['total_episodes'] = episode_counter

    print(f"\nMerged {episode_counter} episodes into: {output_file}")
    return episode_counter


def main():
    parser = argparse.ArgumentParser(description='Merge multiple HDF5 files')
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
        sys.exit(1)

    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f}")

    # Verify all files exist
    for f in input_files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)

    merge_hdf5_files(input_files, args.output)


if __name__ == '__main__':
    main()
