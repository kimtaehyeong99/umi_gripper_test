#!/bin/bash
# UMI Data Pipeline - Raw Data Recording Script
# Records RGB, Depth, and Gripper data to ROS2 bag

set -e

# Default session name with timestamp
SESSION_NAME=${1:-"session_$(date +%Y%m%d_%H%M%S)"}

# Get script directory and package path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PACKAGE_DIR}/data/raw"

# Create data directory if not exists
mkdir -p "$DATA_DIR"

# Output path
OUTPUT_PATH="${DATA_DIR}/${SESSION_NAME}"

echo "======================================"
echo "UMI Data Recording"
echo "======================================"
echo "Session: ${SESSION_NAME}"
echo "Output:  ${OUTPUT_PATH}"
echo "======================================"
echo ""
echo "Recording topics:"
echo "  - /camera/camera/color/image_rect_raw (RGB)"
echo "  - /camera/camera/aligned_depth_to_color/image_raw (Depth)"
echo "  - /camera/camera/color/camera_info (Camera Info)"
echo "  - /gripper_position_controller/commands (Gripper)"
echo ""
echo "Press Ctrl+C to stop recording"
echo "======================================"

# Record topics
ros2 bag record -o "$OUTPUT_PATH" \
  /camera/camera/color/image_rect_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/color/camera_info \
  /gripper_position_controller/commands

echo ""
echo "Recording saved to: ${OUTPUT_PATH}"
echo "To view bag info: ros2 bag info ${OUTPUT_PATH}"
