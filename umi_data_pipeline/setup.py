from setuptools import setup, find_packages

package_name = 'umi_data_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'h5py',
        'zarr',
        'scipy',
    ],
    zip_safe=True,
    maintainer='robotis-ai',
    maintainer_email='user@example.com',
    description='UMI Data Pipeline for data collection and conversion',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)
