# UMI Gripper Test - Meta Repository

UMI (Universal Manipulation Interface) 스타일 그리퍼 데이터 수집 및 처리를 위한 메타 레포지토리

## 프로젝트 구조

```
umi_gripper_test/
├── ros2_orb_slam3/                      # ORB-SLAM3 ROS2 래퍼 (서브모듈)
├── dynamixel_hardware_interface_demos/  # Dynamixel 그리퍼 드라이버 (서브모듈)
└── umi_data_pipeline/                   # 데이터 수집/변환 파이프라인
```

## 서브모듈

### ros2_orb_slam3
- **용도**: RGBD SLAM으로 카메라 6DoF 포즈 추정
- **카메라**: RealSense D405 (640x480@30fps)
- **추가 기능**:
  - `/orb_slam3/camera_pose` 토픽으로 포즈 퍼블리시
  - `/orb_slam3/camera_path` 토픽으로 경로 퍼블리시
  - TF2 broadcast (odom → camera_link)
  - `RGBD.MinDepth` 파라미터로 그리퍼 필터링 (0.25m 이내 무시)

### dynamixel_hardware_interface_demos
- **용도**: Dynamixel 기반 그리퍼 제어
- **토픽**: `/trigger_position_controller/commands` (Float64MultiArray)

### umi_data_pipeline
- **용도**: UMI/LeRobot 형식 데이터셋 생성
- **워크플로우**:
  1. 데이터 취득 → ROS2 bag
  2. 데이터 정합 → 오프라인 SLAM → HDF5
  3. 데이터 변환 → Zarr/LeRobot

## 빠른 시작

```bash
# 워크스페이스 빌드
cd ~/umi_ws
colcon build

# 소스
source install/setup.bash

# 데이터 수집 시작
ros2 launch umi_data_pipeline data_collection.launch.py
```

## 관련 문서

- [umi_data_pipeline/README.md](umi_data_pipeline/README.md) - 데이터 파이프라인 상세 사용법
- [ros2_orb_slam3/orb_slam3/config/RGBD/TUNING_GUIDE.md](ros2_orb_slam3/orb_slam3/config/RGBD/TUNING_GUIDE.md) - SLAM 파라미터 튜닝 가이드

## 하드웨어 구성

| 장치 | 모델 | 설정 |
|------|------|------|
| 카메라 | RealSense D405 | 640x480@30fps, USB 2.1 |
| 그리퍼 | Dynamixel 기반 커스텀 | 토픽: `/trigger_position_controller/commands` |

## TODO

- [ ] `process_bag_with_slam.py`에 실제 ORB-SLAM3 오프라인 처리 연동
- [ ] 그리퍼 캘리브레이션 자동화
- [ ] 에피소드 자동 분할 기능
