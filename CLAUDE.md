# Claude Code Context - UMI Gripper Test

이 파일은 Claude Code가 프로젝트를 이해하는 데 도움을 주기 위한 컨텍스트 파일입니다.

## 프로젝트 개요

UMI (Universal Manipulation Interface) 스타일 로봇 그리퍼 데이터 수집 시스템

## 현재 상태 (2024-02)

### 완료된 작업

1. **메타 레포지토리 구성**
   - Git 서브모듈로 `ros2_orb_slam3`, `dynamixel_hardware_interface_demos` 연결

2. **ros2_orb_slam3 수정 사항** (서브모듈)
   - RealSense D405 설정 파일 추가 (`config/RGBD/RealSense_D405.yaml`)
   - `RGBD.MinDepth` 파라미터 추가 (그리퍼 필터링용)
   - 포즈 퍼블리셔 추가 (`/orb_slam3/camera_pose`, `/orb_slam3/camera_path`)
   - TF2 broadcast 추가 (odom → camera_link)
   - SLAM 파라미터 튜닝 가이드 작성

3. **umi_data_pipeline 패키지 생성**
   - 3단계 워크플로우 구현:
     - 1단계: `record_raw_data.sh` - ROS2 bag 레코딩
     - 2단계: `process_bag_with_slam.py` - SLAM + HDF5 변환 (placeholder)
     - 3단계: `convert_to_umi_zarr.py` - Zarr 변환

### 진행 중 / TODO

1. **오프라인 SLAM 연동**
   - `process_bag_with_slam.py`에 실제 ORB-SLAM3 연동 필요
   - 현재는 placeholder 포즈 (identity) 생성

2. **그리퍼 캘리브레이션**
   - Dynamixel 모터 위치 → 실제 그리퍼 너비(m) 매핑
   - `config/recording_config.yaml`에서 설정

3. **에피소드 분할**
   - 현재 전체를 하나의 에피소드로 처리
   - 자동/수동 분할 기능 필요

## 하드웨어 환경

| 장치 | 모델 | 연결 | 비고 |
|------|------|------|------|
| 카메라 | RealSense D405 | USB 2.1 | 640x480@30fps |
| 그리퍼 | Dynamixel 커스텀 | USB | 토픽: `/trigger_position_controller/commands` |

## 주요 파일 위치

```
umi_gripper_test/
├── ros2_orb_slam3/
│   ├── src/common.cpp                    # SLAM 노드 + 포즈 퍼블리셔
│   ├── include/ros2_orb_slam3/common.hpp
│   └── orb_slam3/config/RGBD/
│       ├── RealSense_D405.yaml           # 카메라 설정
│       └── TUNING_GUIDE.md               # 튜닝 가이드
│
├── umi_data_pipeline/
│   ├── scripts/
│   │   ├── record_raw_data.sh            # 1단계
│   │   ├── process_bag_with_slam.py      # 2단계 (TODO: SLAM 연동)
│   │   └── convert_to_umi_zarr.py        # 3단계
│   ├── config/recording_config.yaml      # 토픽, 캘리브레이션 설정
│   └── data/                             # 데이터 저장 위치
│
└── dynamixel_hardware_interface_demos/   # 그리퍼 드라이버
```

## 빌드 및 실행

```bash
# 빌드
cd ~/umi_ws
colcon build

# 소스
source install/setup.bash

# SLAM 실행 (실시간)
ros2 run ros2_orb_slam3 rgbd_node_cpp

# 데이터 수집
ros2 launch umi_data_pipeline data_collection.launch.py
./scripts/record_raw_data.sh session_name
```

## 참조 레포지토리

- **UMI 원본**: `/home/robotis-ai/umi_ws/src/universal_manipulation_interface`
- **데이터 형식**: Zarr (.zarr.zip), UMI diffusion policy 호환

## 다음 작업 제안

1. `process_bag_with_slam.py`에 실제 SLAM 연동
   - Option A: ros2 bag play + SLAM 노드 실행 후 trajectory 파일 읽기
   - Option B: Python에서 직접 이미지 처리 + SLAM 라이브러리 호출

2. 그리퍼 캘리브레이션 스크립트 작성

3. 실제 데이터 수집 테스트
