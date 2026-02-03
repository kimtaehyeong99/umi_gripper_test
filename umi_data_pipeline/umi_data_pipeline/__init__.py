"""
UMI Data Pipeline

A ROS2 package for collecting and converting robot manipulation data
into UMI/LeRobot compatible formats.

Workflow:
1. Data Acquisition: RGB + Depth + Gripper -> ROS2 bag
2. Data Processing: ROS2 bag -> Offline SLAM -> HDF5
3. Data Conversion: HDF5 -> Zarr/LeRobot
"""

__version__ = '0.1.0'
