#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER_NAME="umi_gripper"

show_help() {
    echo "UMI Gripper Test - Docker Manager"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Infrastructure:"
    echo "  build     Build Docker image"
    echo "  start     Start container"
    echo "  enter     Enter running container"
    echo "  stop      Stop container"
    echo "  setup     First-time: build all ROS2 packages inside container"
    echo ""
    echo "Workflow:"
    echo "  record    Data collection (gripper + camera bringup)"
    echo "  train     Start training"
    echo "  infer     Start inference server (GPU)"
    echo "  deploy    Start bridge + motion controller"
    echo ""
    echo "  help      Show this help"
}

_ensure_running() {
    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        echo "Error: Container not running. Run '$0 start' first."
        exit 1
    fi
}

_setup_x11() {
    if [ -n "$DISPLAY" ]; then
        xhost +local:docker 2>/dev/null || true
    fi
}

build_image() {
    echo "Building umi_gripper Docker image..."
    docker compose -f "${SCRIPT_DIR}/docker-compose.yml" build
}

start_container() {
    _setup_x11
    echo "Starting umi_gripper container..."
    docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d
    echo "Container started. Use '$0 enter' to open a shell."
}

enter_container() {
    _setup_x11
    _ensure_running
    docker exec -it "$CONTAINER_NAME" bash
}

stop_container() {
    docker compose -f "${SCRIPT_DIR}/docker-compose.yml" down
    echo "Container stopped."
}

setup_packages() {
    _ensure_running
    echo "Building ROS2 packages inside container..."
    docker exec "$CONTAINER_NAME" bash -c '
        source /opt/ros/jazzy/setup.bash
        source /root/ros2_ws/install/setup.bash 2>/dev/null
        cd /root/ros2_ws

        # Symlink volume-mapped packages into workspace
        for pkg in umi_policy_bridge robotis_motion_controller umi_data_pipeline ros2_orb_slam3 dynamixel_hardware_interface_demos; do
            src="/root/ros2_ws/src/umi_gripper_test/$pkg"
            if [ -d "$src" ] && [ ! -L "/root/ros2_ws/src/$pkg" ]; then
                ln -sf "$src" "/root/ros2_ws/src/$pkg"
                echo "Linked: $pkg"
            fi
        done

        # Build all workspace packages
        colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
        echo ""
        echo "=== Setup complete ==="
    '
}

start_record() {
    _ensure_running
    echo "Starting data collection..."
    docker exec -it "$CONTAINER_NAME" bash -c '
        source /root/ros2_ws/install/setup.bash
        echo "=== UMI Data Collection ==="
        echo "Step 1: Launch gripper + camera in separate terminals:"
        echo "  ros2 launch dynamixel_hardware_interface_example_4 hardware_dual.launch.py"
        echo "  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30"
        echo ""
        echo "Step 2: Record a demo:"
        echo "  cd /root/ros2_ws/src/umi_gripper_test/umi_data_pipeline"
        echo "  ./scripts/record_raw_data.sh data/raw/demo_XX"
        echo ""
        exec bash
    '
}

start_train() {
    _ensure_running
    echo "Starting training..."
    docker exec -it "$CONTAINER_NAME" bash -c '
        source /root/ros2_ws/install/setup.bash
        export PYTHONPATH=/root/ros2_ws/src/umi_gripper_test/detached-umi-policy:${PYTHONPATH}
        export WANDB_MODE=disabled
        export HF_HUB_OFFLINE=1
        cd /root/ros2_ws/src/umi_gripper_test/detached-umi-policy
        echo "=== Training ==="
        echo "Usage: python train.py --config-name=train_diffusion_unet_timm_umi_workspace \\"
        echo "  task.dataset_path=/root/ros2_ws/src/umi_gripper_test/umi_data_pipeline/data/zarr/merged.zarr.zip \\"
        echo "  training.num_epochs=200"
        echo ""
        exec bash
    '
}

start_inference() {
    _ensure_running
    local ckpt="${1:-}"
    if [ -z "$ckpt" ]; then
        echo "Usage: $0 infer <checkpoint_path>"
        echo "Example: $0 infer data/outputs/2026.02.10/18.50.33_.../checkpoints/epoch=0060-train_loss=0.024.ckpt"
        exit 1
    fi
    echo "Starting inference server..."
    docker exec -it "$CONTAINER_NAME" bash -c "
        export PYTHONPATH=/root/ros2_ws/src/umi_gripper_test/detached-umi-policy:\${PYTHONPATH}
        export HF_HUB_OFFLINE=1
        cd /root/ros2_ws/src/umi_gripper_test/detached-umi-policy
        python3 detached_policy_inference.py -i ${ckpt} --port 8766
    "
}

start_deploy() {
    _ensure_running
    echo "Starting bridge + motion controller..."
    docker exec -it "$CONTAINER_NAME" bash -c '
        source /root/ros2_ws/install/setup.bash
        ros2 launch umi_policy_bridge umi_policy_bridge.launch.py zmq_host:=127.0.0.1
    '
}

clean_build() {
    _ensure_running
    echo "Cleaning build cache..."
    docker exec "$CONTAINER_NAME" bash -c "rm -rf /root/ros2_ws/build/* /root/ros2_ws/install/* /root/ros2_ws/log/*"
    echo "Cleaned. Run '$0 setup' to rebuild."
}

case "$1" in
    "build")   build_image ;;
    "start")   start_container ;;
    "enter")   enter_container ;;
    "stop")    stop_container ;;
    "setup")   setup_packages ;;
    "clean")   clean_build ;;
    "record")  start_record ;;
    "train")   start_train ;;
    "infer")   shift; start_inference "$@" ;;
    "deploy")  start_deploy ;;
    "help")    show_help ;;
    *)         show_help; exit 1 ;;
esac
