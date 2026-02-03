# UMI Data Pipeline

RealSense D405 + Dynamixel 그리퍼 환경에서 UMI/LeRobot 호환 데이터셋을 생성하는 ROS2 패키지

## 워크플로우 (3단계)

```
[1단계: 데이터 취득]
RGB + Depth + Gripper → ROS2 bag (raw 데이터)

[2단계: 데이터 정합]
ROS2 bag → 오프라인 SLAM → HDF5

[3단계: 데이터 변환]
HDF5 → Zarr (UMI) / LeRobot
```

## 설치

### 의존성

```bash
# Python 패키지
pip install rosbags zarr h5py scipy imageio opencv-python

# ROS2 빌드
cd ~/umi_ws
colcon build --packages-select umi_data_pipeline
source install/setup.bash
```

## 사용법

### 1단계: 데이터 취득

```bash
# 터미널 1: 카메라 + 그리퍼 실행
ros2 launch umi_data_pipeline data_collection.launch.py

# 터미널 2: 레코딩
cd ~/umi_ws/src/umi_gripper_test/umi_data_pipeline
./scripts/record_raw_data.sh demo_001
# Ctrl+C로 종료
```

**녹화되는 토픽:**
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/camera/camera/color/image_rect_raw` | Image | RGB 이미지 |
| `/camera/camera/aligned_depth_to_color/image_raw` | Image | Depth 이미지 |
| `/camera/camera/color/camera_info` | CameraInfo | 카메라 파라미터 |
| `/trigger_position_controller/commands` | Float64MultiArray | 그리퍼 위치 |

**저장 위치:** `data/raw/demo_001/`

### 2단계: SLAM + HDF5 변환

```bash
python3 scripts/process_bag_with_slam.py \
  --input data/raw/demo_001 \
  --output data/processed/demo_001 \
  --config config/recording_config.yaml
```

**출력 구조:**
```
data/processed/demo_001/
├── rgb/                    # RGB 이미지 시퀀스
├── depth/                  # Depth 이미지 시퀀스
├── camera_trajectory.csv   # SLAM 카메라 궤적
├── metadata.json           # 메타데이터
└── dataset.hdf5            # 통합 HDF5 파일
```

**HDF5 구조:**
```
dataset.hdf5
├── episode_0000/
│   ├── rgb_images        # (T, H, W, 3) uint8
│   ├── depth_images      # (T, H, W) uint16 (mm)
│   ├── camera_pose       # (T, 7) [x,y,z,qx,qy,qz,qw]
│   ├── gripper_width     # (T,) float32 (meters)
│   └── timestamps        # (T,) float64
└── metadata/
```

### 3단계: 최종 형식 변환

**UMI Zarr 형식:**
```bash
python3 scripts/convert_to_umi_zarr.py \
  --input data/processed/demo_001/dataset.hdf5 \
  --output data/datasets/umi_demo.zarr.zip \
  --image-size 224 224
```

**LeRobot 형식:**
```bash
python3 scripts/convert_to_lerobot.py \
  --input data/processed/demo_001/dataset.hdf5 \
  --output data/datasets/lerobot_demo/
```

## UMI Zarr 출력 형식

```
dataset.zarr.zip/
├── data/
│   ├── robot0_eef_pos           # (N, 3) 위치
│   ├── robot0_eef_rot_axis_angle # (N, 6) 6D rotation
│   ├── robot0_gripper_width     # (N, 1) 그리퍼 너비
│   ├── camera0_rgb              # (N, 3, 224, 224) RGB
│   └── action                   # (N, 10) 액션
└── meta/
    └── episode_ends             # 에피소드 경계
```

## 설정 파일

`config/recording_config.yaml`:
```yaml
camera:
  rgb_topic: "/camera/camera/color/image_rect_raw"
  depth_topic: "/camera/camera/aligned_depth_to_color/image_raw"
  fps: 30

gripper:
  topic: "/trigger_position_controller/commands"
  min_width: 0.0    # 완전 닫힘 (m)
  max_width: 0.08   # 완전 열림 (m)

output:
  image_size: [224, 224]
```

## 파일 구조

```
umi_data_pipeline/
├── scripts/
│   ├── record_raw_data.sh        # 1단계
│   ├── process_bag_with_slam.py  # 2단계
│   ├── convert_to_umi_zarr.py    # 3단계 (UMI)
│   └── convert_to_lerobot.py     # 3단계 (LeRobot)
├── launch/
│   └── data_collection.launch.py
├── config/
│   └── recording_config.yaml
├── data/                         # 데이터 저장 (gitignore)
│   ├── raw/                      # bag 파일
│   ├── processed/                # HDF5 파일
│   └── datasets/                 # Zarr/LeRobot
└── umi_data_pipeline/            # Python 모듈
    └── __init__.py
```

## 검증

```bash
# bag 파일 확인
ros2 bag info data/raw/demo_001/

# HDF5 확인
python3 -c "
import h5py
with h5py.File('data/processed/demo_001/dataset.hdf5', 'r') as f:
    print(list(f.keys()))
"

# Zarr 확인
python3 -c "
import zarr
z = zarr.open('data/datasets/umi_demo.zarr.zip')
print(z.tree())
"
```

## 주의사항

1. **SLAM**: 현재 `process_bag_with_slam.py`는 placeholder 포즈를 생성함. 실제 ORB-SLAM3 연동 필요
2. **그리퍼 캘리브레이션**: `recording_config.yaml`에서 `min_position/max_position` → `min_width/max_width` 매핑 설정 필요
3. **에피소드 분할**: 현재 전체를 하나의 에피소드로 처리. 수동/자동 분할 기능 추가 예정
