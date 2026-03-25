#!/bin/bash
# Auto-create symlinks for volume-mapped packages on container start

# Third-party packages
for pkg in dynamixel_hardware_interface_demos ros2_orb_slam3 robotis_motion_controller; do
    src="/root/ros2_ws/src/umi_gripper_test/third_party/$pkg"
    link="/root/ros2_ws/src/$pkg"
    if [ -d "$src" ] && [ ! -L "$link" ]; then
        ln -sf "$src" "$link"
    fi
done

# Wrapper packages
for pkg in umi_bringup umi_data_manager umi_model_manager; do
    src="/root/ros2_ws/src/umi_gripper_test/$pkg"
    link="/root/ros2_ws/src/$pkg"
    if [ -d "$src" ] && [ ! -L "$link" ]; then
        ln -sf "$src" "$link"
    fi
done

# Fix setuptools if needed
pip install -q setuptools==75.8.2 2>/dev/null

# RMW / Zenoh setup
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_DOMAIN_ID=13

# Start Zenoh daemon if not already running
source /opt/ros/jazzy/setup.bash
if ! pgrep -f rmw_zenohd > /dev/null 2>&1; then
    ros2 run rmw_zenoh_cpp rmw_zenohd &
    sleep 1
fi

exec "$@"
