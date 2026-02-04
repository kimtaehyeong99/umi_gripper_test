사용 방법

그리퍼 on
ros2 launch dynamixel_hardware_interface_example_4 hardware_dual.launch.py

리얼센스 on
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true depth_module.depth_profile:=640x480x30 depth_module.color_profile:=640x480x30

데모 레코드
./record_raw_data.sh demo_001

슬램, 데이터 변환
python3 process_bag_with_slam.py   --input ../data/raw/demo_001   --output ../data/processed/demo_001

umi데이터 변환
python3 convert_to_umi_zarr.py   --input ../data/processed/demo_001/dataset.hdf5   --output ../data/zarr/demo_001.zarr.zip

학습
PYTHONPATH=/home/robotis-ai/umi_ws/src/detached-umi-policy:$PYTHONPATH \
WANDB_MODE=disabled \
python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/home/robotis-ai/umi_ws/src/umi_gripper_test/umi_data_pipeline/data/zarr/demo_001.zarr.zip \
  logging.mode=disabled \
  training.num_epochs=120