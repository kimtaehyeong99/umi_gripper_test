FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# ── System packages ──
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    vim \
    git \
    wget \
    # ROS2 middleware (Zenoh for cross-container communication)
    ros-${ROS_DISTRO}-rmw-zenoh-cpp \
    # ROS2 packages for motion_controller
    ros-${ROS_DISTRO}-pinocchio \
    ros-${ROS_DISTRO}-osqp-vendor \
    ros-${ROS_DISTRO}-interactive-markers \
    # ROS2 packages for ORB-SLAM3 & data pipeline
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-compressed-depth-image-transport \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    # ROS2 packages for dynamixel & hardware
    ros-${ROS_DISTRO}-controller-manager \
    ros-${ROS_DISTRO}-joint-state-broadcaster \
    ros-${ROS_DISTRO}-position-controllers \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-joint-state-publisher \
    # C++ build dependencies
    libeigen3-dev \
    libboost-all-dev \
    libssl-dev \
    libglew-dev \
    libepoxy-dev \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    python3-natsort \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.break-system-packages true && \
    pip install setuptools==75.8.2

# ── Pangolin (ORB-SLAM3 visualization dependency) ──
RUN git clone --depth 1 https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin && \
    cd /tmp/Pangolin && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/Pangolin

# ── OsqpEigen (motion_controller dependency) ──
# osqp is provided by ros-jazzy-osqp-vendor at /opt/ros/jazzy
RUN git clone https://github.com/robotology/osqp-eigen.git /tmp/osqp-eigen && \
    cd /tmp/osqp-eigen && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DCMAKE_BUILD_TYPE=Release \
             -Dosqp_DIR=/opt/ros/jazzy/lib/cmake/osqp && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/osqp-eigen

# ── Intel RealSense SDK (librealsense2 from source for Noble) ──
RUN apt-get update && apt-get install -y \
    libusb-1.0-0-dev libglfw3-dev libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 -b v2.56.4 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense && \
    cd /tmp/librealsense && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
             -DBUILD_PYTHON_BINDINGS=ON \
             -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/librealsense

# ── ROS2 RealSense wrapper ──
ENV COLCON_WS=/root/ros2_ws
WORKDIR ${COLCON_WS}
RUN mkdir -p ${COLCON_WS}/src && \
    cd ${COLCON_WS}/src && \
    git clone -b r/4.56.4 https://github.com/IntelRealSense/realsense-ros.git && \
    cd realsense-ros && git checkout cbd8b8487d4c0126bb590822db9469414496e62a

# ── Dynamixel packages ──
RUN cd ${COLCON_WS}/src && \
    git clone -b jazzy https://github.com/ROBOTIS-GIT/dynamixel_hardware_interface.git && \
    git clone -b jazzy https://github.com/ROBOTIS-GIT/dynamixel_interfaces.git && \
    git clone -b jazzy https://github.com/ROBOTIS-GIT/DynamixelSDK.git

# ── Build external ROS2 packages (not volume-mapped) ──
RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd ${COLCON_WS} && \
    rosdep update && \
    rosdep install -i --from-path src --rosdistro ${ROS_DISTRO} \
        --skip-keys='librealsense2 ORB_SLAM3 Pangolin orb_slam3 launch_pytest' -y || true && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# ── PyTorch + CUDA ──
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ── Fix setuptools (PyTorch upgrades it to 81+ which breaks colcon) ──
RUN pip install setuptools==75.8.2

# ── Python dependencies (inference + data pipeline + training) ──
RUN pip install --no-cache-dir --ignore-installed \
    numpy scipy \
    pyzmq \
    opencv-python-headless \
    hydra-core omegaconf dill click \
    "timm==1.0.3" "diffusers==0.27.2" "huggingface_hub<0.24" transformers einops \
    accelerate pytorch_lightning robomimic \
    h5py "zarr<3" imagecodecs numcodecs \
    threadpoolctl filelock \
    pandas wandb \
    rosbags

# ── Shell setup ──
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo '[ -f ${COLCON_WS}/install/setup.bash ] && source ${COLCON_WS}/install/setup.bash' >> ~/.bashrc && \
    echo "alias cb='cd ${COLCON_WS} && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc && \
    echo "export HF_HUB_OFFLINE=1" >> ~/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_zenoh_cpp" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=13" >> ~/.bashrc && \
    echo "alias zenoh='ros2 run rmw_zenoh_cpp rmw_zenohd'" >> ~/.bashrc

# ── Entrypoint (auto-creates symlinks on container start) ──
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR ${COLCON_WS}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
