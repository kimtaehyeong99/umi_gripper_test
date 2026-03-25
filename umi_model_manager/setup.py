from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'umi_model_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotis-ai',
    maintainer_email='robotis@robotis.com',
    description='UMI model training, inference, and robot deployment',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bridge_node = umi_model_manager.bridge_node:main',
            'init_pose = umi_model_manager.init_pose_node:main',
            'visualizer = umi_model_manager.visualizer:main',
        ],
    },
)
