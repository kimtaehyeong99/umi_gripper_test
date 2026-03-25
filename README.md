# UMI Gripper Test

UMI(Universal Manipulation Interface) 기반 그리퍼 조작 학습 및 배포 파이프라인.

## 프로젝트 구조

```
umi_gripper_test/
├── umi_bringup/           # 하드웨어 브링업 (그리퍼 + 카메라)
├── umi_data_manager/      # 데이터 취득, SLAM, 변환
├── umi_model_manager/     # 학습, 추론, 로봇 배포
├── third_party/           # 외부 의존 패키지
│   ├── dynamixel_hardware_interface_demos/
│   ├── ros2_orb_slam3/
│   ├── robotis_motion_controller/
│   └── detached-umi-policy/
└── docker/                # Docker 환경
    ├── Dockerfile
    ├── docker-compose.yml
    ├── container.sh       # Docker 관리 스크립트
    ├── entrypoint.sh
    └── workspace/         # 영구 볼륨 (data, build)
```

## 초기 설정

```bash
# 1. Docker 이미지 빌드
./docker/container.sh build

# 2. 컨테이너 시작
./docker/container.sh start

# 3. ROS2 패키지 빌드 (최초 1회)
./docker/container.sh setup

# 4. 컨테이너 접속
./docker/container.sh enter
```

### container.sh 명령어

| 명령어 | 설명 |
|--------|------|
| `build` | Docker 이미지 빌드 |
| `start` | 컨테이너 시작 |
| `stop` | 컨테이너 정지 |
| `enter` | 컨테이너 접속 |
| `setup` | colcon build (최초 1회) |
| `clean` | 빌드 캐시 삭제 |

## 워크플로우

### 1. 하드웨어 브링업

```bash
# 터미널 1: 그리퍼 + 카메라 시작 (계속 켜놓음)
ros2 launch umi_bringup umi_bringup.launch.py
```

### 2. 데이터 취득

```bash
# 터미널 2: 에피소드별 녹화 (Ctrl+C로 종료)
ros2 launch umi_data_manager umi_record.launch.py session:=demo_01
# 다시 실행하면 episode_002 자동 생성
ros2 launch umi_data_manager umi_record.launch.py session:=demo_01
```

데이터 저장 경로: `/workspace/data/raw/demo_01/episode_001/`

### 3. SLAM + 데이터 변환

```bash
# bringup 종료 후 실행 (카메라 토픽 충돌 방지)

# 에피소드 하나 처리
cd /root/ros2_ws/src/umi_gripper_test/umi_data_manager
python3 scripts/process_bag_with_slam.py \
  --input /workspace/data/raw/demo_01/episode_001 \
  --output /workspace/data/processed/demo_01/episode_001

# 전체 에피소드 일괄 처리
python3 scripts/batch_process_slam.py \
  -i /workspace/data/raw/demo_01 \
  -o /workspace/data/processed/demo_01
```

### 4. 데이터 머지 + Zarr 변환

```bash
# 여러 에피소드 → 1개 HDF5
python3 scripts/merge_umi_hdf5.py \
  -d /workspace/data/processed/demo_01 \
  -o /workspace/data/processed/demo_01/merged.hdf5

# HDF5 → Zarr (학습용)
python3 scripts/convert_to_umi_zarr.py \
  --input /workspace/data/processed/demo_01/merged.hdf5 \
  --output /workspace/data/zarr/demo_01.zarr.zip
```

### 5. 학습

```bash
export PYTHONPATH=/root/ros2_ws/src/umi_gripper_test/third_party/detached-umi-policy:${PYTHONPATH}
cd /root/ros2_ws/src/umi_gripper_test/third_party/detached-umi-policy

WANDB_MODE=disabled python3 train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/workspace/data/zarr/demo_01.zarr.zip \
  logging.mode=disabled \
  training.num_epochs=200
```

체크포인트 저장 경로: `third_party/detached-umi-policy/data/outputs/<날짜>/<시간>/checkpoints/`

### 6. 추론 서버 시작

```bash
export PYTHONPATH=/root/ros2_ws/src/umi_gripper_test/third_party/detached-umi-policy:${PYTHONPATH}
cd /root/ros2_ws/src/umi_gripper_test/third_party/detached-umi-policy

HF_HUB_OFFLINE=1 python3 detached_policy_inference.py \
  -i <체크포인트 경로> --port 8766
```

### 7. 로봇 배포 (Deploy)

AI Worker 로봇 (별도 Docker)에 배포하는 경우:

```bash
# AI Worker Docker 터미널 1: Gazebo 시뮬레이션
ros2 launch ffw_bringup ffw_sg2_follower_ai_gazebo.launch.py

# AI Worker Docker 터미널 2: Bridge + Motion Controller
ros2 launch umi_model_manager umi_deploy.launch.py

# AI Worker Docker 터미널 3: 초기 자세
ros2 run umi_model_manager init_pose

# AI Worker Docker 터미널 4: 에피소드 시작/종료
ros2 service call /umi_policy_bridge/start_episode std_srvs/srv/Trigger
ros2 service call /umi_policy_bridge/stop_episode std_srvs/srv/Trigger
```

### 주요 설정

**Bridge 설정** (`umi_model_manager/config/bridge_config.yaml`)
- `zmq_host`: 추론 서버 IP (Docker 내부: `172.17.0.1`)
- `camera_topic`: RealSense 카메라 토픽
- `use_cam_frame`: true (SLAM 카메라 좌표계 변환)
- `action_hz`: 30.0 (controller command_hz와 일치)
- `max_pos_delta` / `max_rot_delta`: 안전 제한

**Controller 설정** (`third_party/robotis_motion_controller/motion_controller_ros/config/controller_config.yaml`)
- `relative_pose_topic`: delta pose 수신 토픽
- `command_hz`: 30.0
- `kp_position` / `kp_orientation`: 추적 게인

## 데이터 구조

```
/workspace/data/
├── raw/                    # bag 녹화 원본
│   └── demo_01/
│       ├── episode_001/
│       ├── episode_002/
│       └── ...
├── processed/              # SLAM + HDF5 변환
│   └── demo_01/
│       ├── episode_001/dataset.hdf5
│       ├── episode_002/dataset.hdf5
│       └── merged.hdf5
└── zarr/                   # 학습용 Zarr
    └── demo_01.zarr.zip
```
