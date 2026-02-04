# UMI Data Pipeline

RealSense D405 + Dynamixel 그리퍼 환경에서 UMI(Universal Manipulation Interface) 호환 데이터셋을 생성하는 ROS2 패키지

## 개요

```
[Stage 1: 데이터 녹화]
RGB + Depth + Gripper Commands + Joint States → ROS2 Bag

[Stage 2: SLAM 처리]
ROS2 Bag → ORB-SLAM3 → HDF5 (with camera poses)

[Stage 3: 형식 변환]
HDF5 → UMI Zarr (학습용)
```

## 설치

### 의존성

```bash
# Python 패키지
pip install rosbags zarr h5py scipy imageio opencv-python numcodecs blosc

# ROS2 빌드
cd ~/umi_ws
colcon build --packages-select umi_data_pipeline ros2_orb_slam3
source install/setup.bash
```

## 사용법

### Stage 1: 데이터 녹화

```bash
# 터미널 1: 카메라 + 그리퍼 하드웨어 실행
ros2 launch realsense2_camera rs_launch.py
ros2 launch dynamixel_hardware_interface_demos gripper.launch.py

# 터미널 2: 녹화 시작
cd ~/umi_ws/src/umi_gripper_test/umi_data_pipeline
./scripts/record_raw_data.sh session_01
# Ctrl+C로 종료
```

**녹화 토픽:**
| 토픽 | 용도 |
|------|------|
| `/camera/camera/color/image_rect_raw/compressed` | RGB 이미지 |
| `/camera/camera/aligned_depth_to_color/image_raw/compressedDepth` | Depth 이미지 |
| `/camera/camera/color/camera_info` | 카메라 파라미터 |
| `/gripper_position_controller/commands` | Gripper Action (명령값) |
| `/joint_states` | Gripper Observation (실제 상태) |

**저장 위치:** `data/raw/session_01/`

### Stage 2: SLAM 처리 + HDF5 변환

**최적화 버전 (v2.0)** - 메모리 기반 처리로 50-60% 속도 향상

```bash
# Fast mode (기본): SLAM 없이 빠르게 처리 (placeholder poses)
python3 scripts/process_bag_with_slam.py \
  --input data/raw/session_01 \
  --output data/processed/session_01 \
  --config config/recording_config.yaml

# SLAM mode: ROS2 SLAM 노드로 실제 카메라 pose 추출
python3 scripts/process_bag_with_slam.py \
  --input data/raw/session_01 \
  --output data/processed/session_01 \
  --config config/recording_config.yaml \
  --use-ros2-slam
```

**최적화 내용:**
| 항목 | 이전 | 최적화 후 |
|------|------|----------|
| 이미지 I/O | 디스크 2번 (저장+읽기) | 메모리 직접 처리 |
| 타임스탬프 매칭 | O(N×M) | O(N log M) |
| SLAM 초기화 | 5초 고정 대기 | 스마트 체크 (<1초) |
| 압축 이미지 | 미지원 | 지원 |

**출력:**
```
data/processed/session_01/
├── camera_trajectory.csv   # SLAM 카메라 궤적
├── metadata.json           # 메타데이터
└── dataset.hdf5            # 통합 HDF5 파일
```

### 다중 세션 처리 (여러 bag 파일)

```bash
# 각 세션 처리
python3 scripts/process_bag_with_slam.py --input data/raw/session_01 --output data/processed/session_01
python3 scripts/process_bag_with_slam.py --input data/raw/session_02 --output data/processed/session_02
python3 scripts/process_bag_with_slam.py --input data/raw/session_03 --output data/processed/session_03

# HDF5 파일 병합
python3 scripts/merge_hdf5.py \
  --input-dir data/processed \
  --output data/merged/dataset.hdf5
```

### Stage 3: UMI Zarr 변환

```bash
python3 scripts/convert_to_umi_zarr.py \
  --input data/processed/session_01/dataset.hdf5 \
  --output data/datasets/umi_demo.zarr.zip

# 또는 병합된 파일 사용
python3 scripts/convert_to_umi_zarr.py \
  --input data/merged/dataset.hdf5 \
  --output data/datasets/umi_demo.zarr.zip
```

## 데이터 형식

### HDF5 구조 (중간 형식)

```
dataset.hdf5
├── episode_000/
│   ├── rgb_images        # [T, H, W, 3] uint8
│   ├── depth_images      # [T, H, W] uint16 (mm)
│   ├── camera_pose       # [T, 7] float32 [x,y,z,qx,qy,qz,qw]
│   ├── gripper_width     # [T] float32 - Observation (실제 상태)
│   ├── gripper_action    # [T] float32 - Action (명령값)
│   └── timestamps        # [T] float64
└── metadata/
    └── config            # 설정 정보
```

### UMI Zarr 구조 (학습용)

```
dataset.zarr.zip/
├── data/
│   ├── robot0_eef_pos              # [N, 3] float32 - position (meters)
│   ├── robot0_eef_rot_axis_angle   # [N, 3] float32 - axis-angle rotation
│   ├── robot0_gripper_width        # [N, 1] float32 - gripper width (meters)
│   ├── robot0_demo_start_pose      # [N, 6] float32 - episode start pose
│   ├── robot0_demo_end_pose        # [N, 6] float32 - episode end pose
│   ├── camera0_rgb                 # [N, 224, 224, 3] uint8 - HWC format
│   └── action                      # [N, 10] float32 - pos(3) + rot6d(6) + gripper(1)
└── meta/
    └── episode_ends                # [episodes] int64 - cumulative indices
```

**회전 표현:**
- `robot0_eef_rot_axis_angle`: axis-angle [3] 저장 → 학습 시 rot6d [6] 변환
- `action`: rot6d [6] 직접 저장

**Gripper 분리:**
- `robot0_gripper_width`: Observation (실제 상태, `/joint_states`)
- `action[:, 9]`: Action (명령값, `/gripper_position_controller/commands`)

## 학습

### 환경 설정 (RTX 5090/Blackwell GPU)

```bash
# Python 3.10 + PyTorch nightly (CUDA 12.8) 환경 생성
conda create -n umi310 python=3.10
conda activate umi310
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# detached-umi-policy 의존성 설치
cd ~/umi_ws/src/detached-umi-policy
pip install -e .
```

### 학습 실행

```bash
cd ~/umi_ws/src/detached-umi-policy

PYTHONPATH=/home/robotis-ai/umi_ws/src/detached-umi-policy:$PYTHONPATH \
WANDB_MODE=disabled \
python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/home/robotis-ai/umi_ws/src/umi_gripper_test/umi_data_pipeline/data/datasets/umi_demo.zarr.zip \
  logging.mode=disabled \
  training.num_epochs=120
```

### 학습 모니터링

```bash
# 체크포인트 확인 (loss 추이)
ls ~/umi_ws/src/detached-umi-policy/data/outputs/*/checkpoints/*.ckpt

# GPU 사용량 확인
nvidia-smi
```

## 설정 파일

`config/recording_config.yaml`:
```yaml
camera:
  rgb_topic: "/camera/camera/color/image_rect_raw/compressed"
  depth_topic: "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth"
  camera_info_topic: "/camera/camera/color/camera_info"
  fps: 30

gripper:
  action_topic: "/gripper_position_controller/commands"      # 명령값
  observation_topic: "/joint_states"                         # 실제 상태
  joint_name: "gripper"
  min_position: 0.0
  max_position: 1.0
  min_width: 0.0
  max_width: 0.08

slam:
  vocabulary_path: "orb_slam3/Vocabulary/ORBvoc.txt.bin"
  settings_file: "orb_slam3/config/RGBD/RealSense_D405.yaml"

output:
  image_size: [224, 224]
  fps: 30
```

## 검증

```bash
# Bag 파일 확인
ros2 bag info data/raw/session_01/

# HDF5 확인
python3 -c "
import h5py
with h5py.File('data/processed/session_01/dataset.hdf5', 'r') as f:
    for key in f.keys():
        if key.startswith('episode'):
            print(f'{key}:')
            for k, v in f[key].items():
                print(f'  {k}: {v.shape}')
"

# Zarr 확인
python3 -c "
import zarr
z = zarr.open('data/datasets/umi_demo.zarr.zip')
print(z.tree())
print('eef_rot shape:', z['data/robot0_eef_rot_axis_angle'].shape)  # [N, 3]
print('rgb shape:', z['data/camera0_rgb'].shape)  # [N, 224, 224, 3]
print('action shape:', z['data/action'].shape)  # [N, 10]
"
```

## 파일 구조

```
umi_data_pipeline/
├── scripts/
│   ├── record_raw_data.sh        # Stage 1: 데이터 녹화
│   ├── process_bag_with_slam.py  # Stage 2: SLAM + HDF5
│   ├── merge_hdf5.py             # 다중 HDF5 병합
│   ├── convert_to_umi_zarr.py    # Stage 3: UMI Zarr 변환
│   └── convert_to_lerobot.py     # Stage 3: LeRobot 변환 (optional)
├── launch/
│   └── data_collection.launch.py
├── config/
│   └── recording_config.yaml
├── data/                         # 데이터 저장 (gitignore)
│   ├── raw/                      # ROS2 bag 파일
│   ├── processed/                # HDF5 파일
│   ├── merged/                   # 병합된 HDF5
│   └── datasets/                 # Zarr/LeRobot
└── umi_data_pipeline/
    └── __init__.py
```

## 전체 파이프라인 예시

```bash
# 1. 데이터 녹화 (3개 세션)
./scripts/record_raw_data.sh session_01
./scripts/record_raw_data.sh session_02
./scripts/record_raw_data.sh session_03

# 2. 각 세션 처리 (SLAM 포함)
for i in 01 02 03; do
  python3 scripts/process_bag_with_slam.py \
    --input data/raw/session_$i \
    --output data/processed/session_$i
done

# 3. HDF5 병합
python3 scripts/merge_hdf5.py \
  --input-dir data/processed \
  --output data/merged/dataset.hdf5

# 4. UMI Zarr 변환
python3 scripts/convert_to_umi_zarr.py \
  --input data/merged/dataset.hdf5 \
  --output data/datasets/umi_demo.zarr.zip

# 5. 학습
cd ~/umi_ws/src/detached-umi-policy
conda activate umi310
PYTHONPATH=$PWD:$PYTHONPATH python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=~/umi_ws/src/umi_gripper_test/umi_data_pipeline/data/datasets/umi_demo.zarr.zip
```
