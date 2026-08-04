# Octo UR5e Fine-tuning

UR5e에서 사람의 프리드라이브 시연을 수집하고, 동일 궤적을 재생하면서 두 카메라와 로봇 상태를 동기 기록한 뒤, Processing → RLDS/TFDS → Octo fine-tuning → 실제 로봇 추론까지 수행하는 end-to-end 연구 파이프라인입니다.

이 저장소가 담당하는 범위는 다음과 같습니다.

```text
프리드라이브 시연 기록
  → 관절 궤적 Replay + Primary/Wrist 영상 및 상태 기록
  → ROS timestamp 기반 10 Hz 동기화와 7D action 생성
  → episode 단위 train/validation RLDS 생성
  → Octo-Small fine-tuning
  → 안전 제한이 적용된 UR5e inference
```

> **안전 주의**
>
> 실제 로봇을 움직이는 명령은 작업 공간을 비우고, UR 속도를 낮추고, 비상 정지 장치와 teach pendant를 즉시 사용할 수 있는 상태에서만 실행하십시오. 이 프로젝트는 로봇의 전원, 브레이크, PolyScope 프로그램을 자동으로 관리하지 않습니다. `--execute`가 없는 주요 명령은 dry-run이지만, `--execute`를 붙이면 실제 그리퍼 또는 로봇이 움직일 수 있습니다. MoveIt 충돌 검사는 planning scene에 등록되지 않은 테이블·트레이·물체를 알 수 없습니다.

## 1. 프로젝트 구조

```text
octo_ur5e_finetuning/
├── collector/
│   ├── config/
│   │   ├── collector.yaml        # Pickup 수집 설정, data/raw 사용
│   │   └── pick_place.yaml       # Pick & Place 설정, data/pick_place/raw 사용
│   └── octo_ur5e_collector/
│       ├── collector_cli.py      # doctor/record/replay/inspect CLI
│       ├── *_node.py             # ROS 2 record/replay/wrist camera 노드
│       ├── core/                 # 궤적, 동기화, 품질, smoothing 로직
│       └── ros_adapters/         # rosbag, controller, camera, gripper 어댑터
├── config/camera/                # RealSense 및 oCam 고정 설정
├── processing/                   # 10 Hz 동기화, action 생성, 품질 평가
├── rlds_builder/                 # TFDS/RLDS planning 및 builder
├── training/                     # dataset 검사, 학습 config, resume, 그래프
├── inference/                    # Octo worker, 실행기, safety/debouncing
├── tests/                        # collector/processing/RLDS/inference 테스트
├── docs/                         # Collector 세부 설계 및 참고 자료
├── docker/                       # 선택적 학습 컨테이너 골격
├── data/
│   ├── raw/                      # Pickup raw episode
│   ├── rlds/                     # Pickup RLDS
│   └── pick_place/
│       ├── raw/                  # Pick & Place raw + processed episode
│       └── rlds/                 # Pick & Place RLDS
├── data_backup/                  # 보존용 데이터 백업(로컬)
├── runs/
│   ├── octo_ur5e/                # checkpoints/config/statistics/plots
│   ├── logs/                     # background training 로그
│   ├── policy_logs/              # inference JSONL 로그
│   └── wandb/                    # W&B offline run
├── pyproject.toml                # 패키지 및 CLI entry points
└── requirements-processing.txt
```

`data/`, `runs/`, `data_backup/`에는 큰 파일과 로컬 산출물이 들어갑니다. 새 작업을 시작할 때 기존 기준 데이터셋을 수정하지 말고 별도 raw/RLDS 경로와 별도 collector 설정 파일을 사용하십시오.

## 2. 요구 환경

### 로봇/수집 워크스테이션

- Ubuntu와 ROS 2 Jazzy
- Universal Robots ROS 2 Driver 및 `scaled_joint_trajectory_controller`
- UR5e PolyScope의 External Control 프로그램
- MoveIt 2 및 `ur_moveit_config` (`--move-to-fixed-start`, 실제 inference에 필요)
- rosbag2 MCAP storage plugin
- FFmpeg (`h264_nvenc` 사용 가능 시 우선, 아니면 `libx264` fallback)
- `v4l2-ctl`을 제공하는 `v4l-utils` (oCam wrist camera)
- Python 3.10 이상
- Primary RGB camera: 현재 `/camera/camera/color/image_raw`
- Wrist camera: WITHROBOT oCam-1CGN-U-T2, GRBG Bayer

### 학습 워크스테이션

- NVIDIA GPU 및 동작하는 CUDA/JAX 환경
- 별도로 clone한 공식 Octo 저장소. 기본 경로는 `/home/sixr/Desktop/octo`
- Conda 환경 `octo_env`에 Octo/JAX/Flax/Optax와 공식 fine-tuning 의존성 설치
- TensorFlow/TFDS는 RLDS 생성과 검사에 필요

ROS 2 Python은 시스템 Python 버전에 종속되므로, 이 프로젝트에서는 역할을 분리합니다.

- `.venv`: collector, camera, processing, RLDS 및 ROS 명령
- `octo_env`: 공식 Octo fine-tuning과 model worker

프롬프트에 `(.venv)`와 `(octo_env)`가 동시에 보이더라도 실제 학습은 `octo_env`의 Python을 명시적으로 사용합니다.

## 3. 설치

### 3.1 프로젝트 설치

```bash
cd /home/sixr/octo_ur5e_finetuning
source /opt/ros/jazzy/setup.bash

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rlds]'
```

ROS가 필요 없는 Processing 전용 환경이라면 최소 설치도 가능합니다.

```bash
python -m pip install -r requirements-processing.txt
python -m pip install -e .
```

설치를 확인합니다.

```bash
octo-collector --version
octo-collector --help
octo-dataset --help
pytest -q
```

### 3.2 Octo 학습 환경

공식 Octo 저장소의 설치 지침에 따라 `octo_env`를 만든 뒤 최소한 다음 import가 성공해야 합니다.

```bash
conda activate octo_env
python -c 'import jax, flax, optax, octo; print(jax.devices())'
```

Octo 저장소가 기본 경로와 다르면 명령마다 `--octo-repo`를 지정하거나 다음 환경 변수를 사용합니다.

```bash
export OCTO_REPO=/path/to/octo
export OCTO_CONDA_ENV=octo_env
```

### 3.3 ROS/카메라 시작 전 확인

UR driver, PolyScope External Control, Primary camera를 먼저 실행하십시오. 실제 실행에 MoveIt이 필요한 경우 별도 터미널에서 다음을 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e launch_rviz:=false
```

Wrist oCam은 stable device link를 기본값으로 사용합니다.

```bash
cd /home/sixr/octo_ur5e_finetuning
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate

octo-wrist-camera
```

필요하면 장치를 명시합니다.

```bash
octo-wrist-camera \
  --device /dev/v4l/by-id/usb-WITHROBOT_Inc._oCam-1CGN-U-T2_SN_3AA01020-video-index0 \
  --width 1280 --height 800 --fps 30
```

카메라가 열리지 않으면 stable link와 점유 프로세스를 확인합니다.

```bash
readlink -f /dev/v4l/by-id/usb-WITHROBOT_Inc._oCam-1CGN-U-T2_SN_3AA01020-video-index0
ls -l /dev/video6
fuser -v /dev/video6
```

주요 토픽은 다음과 같습니다.

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /wrist_camera/image_raw
ros2 topic echo /io_and_status_controller/robot_program_running --once
ros2 control list_controllers
```

Wrist raw topic은 `bayer_grbg8`이고 replay recorder가 한 번 debayer하여 H.264로 저장합니다. `/wrist_camera/image_color`는 preview이며 학습 데이터로 중복 기록하지 않습니다.

## 4. 설정 파일

기본 Pickup 설정은 `collector/config/collector.yaml`, Pick & Place 파일럿 설정은 `collector/config/pick_place.yaml`입니다.

반드시 확인할 항목:

- `robot.base_frame`, `robot.tcp_frame`, `robot.joint_names`
- `ros.*`: joint/TF/controller/IO topic과 action/service 이름
- `cameras`: Primary/Wrist image topic
- `recording_start.tcp_pose6`: `[x, y, z, rx, ry, rz]`, m/rad
- `gripper.output_pin`: 현재 DO0
- `gripper.output_value_for_open/closed`: 현재 open=1, closed=0
- `storage.output_root`: task별 raw 경로
- `synchronization.*`: camera/state/TCP age 허용치
- `action_contract`: 7D action 정의와 normalization mask

DO pin과 polarity를 코드에 하드코딩하지 말고 YAML에서 관리하십시오. `recording_start`를 변경하려면 현재 TCP와 좌표계가 정확한지 확인한 후 별도 task config에 기록하십시오.

## 5. 데이터 수집

### 5.1 Preflight

수집 전 read-only 점검:

```bash
octo-collector doctor \
  --config collector/config/collector.yaml \
  --freedrive
```

Replay 전 점검:

```bash
octo-collector doctor \
  --config collector/config/collector.yaml \
  --replay
```

다음을 모두 확인하십시오.

- safety mode가 normal
- PolyScope External Control 프로그램이 Play 상태
- `/scaled_joint_trajectory_controller/follow_joint_trajectory` 사용 가능
- 양 카메라 토픽 수신
- SetIO service 사용 가능
- 충분한 디스크 여유 공간

### 5.2 시연 기록

Pickup 예시:

```bash
octo-collector record-demo \
  --config collector/config/collector.yaml \
  --instruction "pick up the blue object" \
  --move-to-fixed-start \
  --enable-freedrive \
  --return-to-start \
  --return-to-start-duration-sec 8 \
  --execute
```

Pick & Place 예시:

```bash
octo-collector record-demo \
  --config collector/config/pick_place.yaml \
  --instruction "pick up the blue object and place it in the tray" \
  --move-to-fixed-start \
  --enable-freedrive \
  --return-to-start \
  --return-to-start-duration-sec 8 \
  --execute
```

기록 중 키:

| 키 | 의미 |
|---|---|
| `0` 또는 `o` | 그리퍼 open, semantic state `0` |
| `1` 또는 `c` | 그리퍼 close, semantic state `1` |
| `q` | 정상 종료 및 저장 |
| `Esc` | abort. 가능한 raw 자료는 보존 |

Collector는 시작 시 그리퍼를 열고 설정된 settle 시간(현재 5초)을 기다립니다. `--move-to-fixed-start`는 MoveIt의 collision-aware IK로 설정 TCP pose에 대응하는 joint goal을 구한 뒤 이동합니다. 다만 IK target 검사와 전체 보간 경로 계획은 같은 개념이 아니므로 실제 경로의 안전은 운영자가 확인해야 합니다.

`--enable-freedrive`는 `scaled_joint_trajectory_controller`와 `freedrive_mode_controller`를 전환합니다. 일부 UR/PolyScope 구성에서는 전환 후 External Control 프로그램이 정지할 수 있습니다. record와 replay 사이에 PolyScope Play 상태와 scaled controller 활성 상태를 다시 확인하십시오.

### 5.3 기록 검증과 조회

```bash
EPISODE_PATH=data/raw/<episode_id>

octo-collector validate-demo "$EPISODE_PATH"
octo-collector inspect "$EPISODE_PATH"
```

Pick & Place의 가장 최근 episode를 선택하려면:

```bash
EPISODE_PATH=$(find data/pick_place/raw \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
echo "$EPISODE_PATH"
```

## 6. Replay

Replay는 기본적으로 검증/dry-run이며 `--execute`를 붙여야 실제 궤적을 전송합니다.

먼저 dry-run:

```bash
octo-collector replay "$EPISODE_PATH" \
  --config collector/config/pick_place.yaml
```

시작 위치 이동, 실제 replay, 종료 후 출발 위치 복귀:

```bash
octo-collector replay "$EPISODE_PATH" \
  --config collector/config/pick_place.yaml \
  --move-to-start \
  --move-to-start-duration-sec 8 \
  --return-to-start \
  --return-to-start-duration-sec 8 \
  --wall-clock-gripper-fallback \
  --execute
```

현재 실제 장비에서는 controller-feedback 기반 gripper scheduling이 hardware-verified 상태가 아니므로 `--wall-clock-gripper-fallback`을 명시해야 합니다. 로봇 속도 scaling이 1보다 작으면 trajectory의 계획 시간보다 실제 wall time이 길어질 수 있습니다. 정상 종료 기준은 trajectory action result가 수신되고 `error_code=0`인 시점입니다.

선택적 궤적 smoothing:

```bash
octo-collector replay "$EPISODE_PATH" \
  --config collector/config/pick_place.yaml \
  --move-to-start --move-to-start-duration-sec 8 \
  --smooth-trajectory \
  --smoothing-window-sec 0.35 \
  --smoothing-polyorder 3 \
  --gripper-anchor-window-sec 0.5 \
  --return-to-start --return-to-start-duration-sec 8 \
  --wall-clock-gripper-fallback \
  --execute
```

Smoothing은 trajectory timing을 유지하며 gripper transition 주변을 anchor로 보호합니다. 출력되는 velocity/acceleration과 원본 대비 최대 변화를 확인한 뒤 사용하십시오.

Replay 중에는 trajectory 실행 전후 정리와 영상 인코더 종료까지 기록되므로 프로세스의 wall time과 실제 trajectory execution duration이 다릅니다. 데이터셋 Processing은 `replay_state == "executing"` 구간만 사용하여 시작 대기, settle, return-to-start를 제외합니다.

## 7. Processing

Processing은 ROS 없이 실행할 수 있으며, replay 영상 timestamp CSV와 `robot_states.mcap`을 공통 10 Hz timeline에 동기화합니다.

한 episode 처리:

```bash
octo-process-episode "$EPISODE_PATH" --require-wrist
```

전체 raw 디렉터리 처리:

```bash
octo-process-batch data/pick_place/raw \
  --require-wrist \
  --summary data/pick_place/processing_summary.json
```

처리 결과 확인:

```bash
octo-inspect-processed \
  "$EPISODE_PATH/processed/synchronized_episode.npz"

octo-evaluate-processed \
  "$EPISODE_PATH/processed/synchronized_episode.npz" \
  --require-wrist
```

처리 단계:

1. Replay의 `executing` 구간만 선택
2. `primary_timestamps.csv`, `wrist_timestamps.csv`의 ROS header timestamp 사용
3. Primary의 고유 frame을 10 Hz로 선택
4. 선택된 실제 Primary timestamp에 joint/TCP/gripper를 보간 또는 정렬
5. camera/state age와 synchronization quality 평가
6. 연속 valid 구간을 `segment_id`로 분리
7. 현재 TCP/tool frame 기준 7D action 생성

MKV의 nominal FPS timestamp나 단순 frame-index modulo downsampling은 사용하지 않습니다. Invalid transition은 `segment_id=-1`이며, RLDS builder는 나머지 구간을 이어 붙이지 않고 각 contiguous segment를 독립 trajectory로 만듭니다.

Action 형식:

```text
[dx, dy, dz, drx, dry, drz, gripper]
```

- `dx,dy,dz`: 현재 TCP/tool frame 기준 상대 이동, meter
- `drx,dry,drz`: 상대 회전 rotation vector, radian
- `gripper`: 다음 상태의 absolute semantic value, `0=open`, `1=closed`
- action은 observation `t → t+1` transition
- Octo normalization mask: `[true, true, true, true, true, true, false]`

## 8. RLDS/TFDS 생성

### 8.1 상태 확인과 plan

```bash
octo-dataset doctor data/pick_place/raw

octo-dataset plan data/pick_place/raw \
  --include-grades GOOD,WARNING,BAD \
  --require-wrist \
  --val-episodes 1 \
  --output data/pick_place/rlds_plan.json
```

`plan`에서 포함/제외 episode, 각 segment, train/validation 목록을 확인하십시오. Split은 정렬된 episode ID의 SHA-256 ranking으로 결정되며 동일 episode의 segment가 train과 validation에 동시에 들어가지 않습니다.

### 8.2 Processing과 RLDS를 한 번에 생성

`build`는 기본 dry-run입니다.

```bash
octo-dataset build data/pick_place/raw \
  --output data/pick_place/rlds/ur5e_pick_place_5ep_1val \
  --include-grades GOOD,WARNING,BAD \
  --require-wrist \
  --val-episodes 1
```

출력을 확인한 뒤 실제 생성:

```bash
octo-dataset build data/pick_place/raw \
  --output data/pick_place/rlds/ur5e_pick_place_5ep_1val \
  --include-grades GOOD,WARNING,BAD \
  --require-wrist \
  --val-episodes 1 \
  --execute
```

기존 non-empty output 디렉터리는 덮어쓰지 않습니다. 설정이나 source가 바뀌어 Processing이 stale이면 재처리되며, 명시적으로 전부 다시 처리할 때만 `--force-process`를 사용하십시오.

생성 결과 검사:

```bash
octo-check-dataset \
  data/pick_place/rlds/ur5e_pick_place_5ep_1val/ur5e_pick/1.0.0
```

## 9. 데이터 구조

### 9.1 Raw episode

```text
data/<task>/raw/<episode_id>/
├── manifest.json                    # episode ID, instruction 등
├── status.json                      # 수집 상태
├── config_resolved.yaml             # 당시 적용된 전체 설정 snapshot
├── demonstration/
│   ├── rosbag2/                     # 원 시연 robot topics
│   ├── events.jsonl                 # gripper/operator event
│   ├── samples.npz                  # 작업용 고주기 samples
│   ├── trajectory.npz               # 시간, joint, TCP, gripper trajectory
│   └── validation.json
├── replay/
│   ├── robot_states.mcap            # replay robot/state topics
│   ├── robot_states_metadata.yaml
│   ├── primary.mkv                  # Primary H.264 video
│   ├── wrist.mkv                    # Wrist H.264 video
│   ├── primary_timestamps.csv       # encoded frame별 ROS timestamp
│   ├── wrist_timestamps.csv
│   ├── events.jsonl                 # gripper scheduled/actual/readback event
│   ├── metadata.json
│   ├── execution_summary.json
│   ├── episode_result.json
│   ├── synchronization_index.csv
│   ├── synchronization_summary.json
│   └── quality_report.json
└── processed/
    ├── synchronized_episode.npz
    ├── synchronization_index.csv
    ├── processing_manifest.yaml
    ├── action_statistics.json
    └── quality_report.json
```

Replay quality에는 trajectory planned/execution/recording/command wall time, joint tracking RMSE/max error, gripper timing, TCP age, camera/state synchronization 지표가 기록됩니다. `evaluation.overall`과 `quality_grade`는 `GOOD`, `WARNING`, `BAD`로 요약됩니다.

### 9.2 Processed NPZ 주요 배열

구체적인 전체 목록과 shape는 `octo-inspect-processed`로 확인하십시오. 핵심 항목은 다음과 같습니다.

- observation timestamp와 source frame index
- `primary_frame_indices`, `wrist_frame_indices`
- `joint_positions`
- `tcp_positions`, `tcp_quaternions_xyzw`
- `gripper_states`
- `actions` (`N × 7`)
- `valid_mask`
- `segment_id`, `segment_is_first`, `segment_is_last`

### 9.3 RLDS step schema

```text
episode
├── episode_metadata
│   ├── episode_id
│   ├── segment_id
│   └── quality_grade
└── steps
    ├── observation
    │   ├── image_primary       uint8 [256, 256, 3]
    │   ├── image_wrist         uint8 [256, 256, 3]
    │   ├── wrist_valid         bool
    │   ├── proprio             float32 [14]
    │   ├── joint_position      float32 [6]
    │   ├── tcp_pose            float32 [7], xyz + quaternion xyzw
    │   └── gripper_state       int64
    ├── action                  float32 [7]
    ├── language_instruction    string
    ├── is_first
    ├── is_last
    └── is_terminal
```

영상은 RLDS build 시 선택된 frame만 decode하고 center square crop 후 256×256 RGB로 resize합니다. Fine-tuning 시 Primary는 256×256, Wrist tokenizer 입력은 128×128로 설정되어 있습니다.

## 10. Fine-tuning

현재 기본 설정 (`training/finetune_config.py`):

- pretrained: `hf://rail-berkeley/octo-small-1.5`
- mode: `head_only`
- language conditioned
- Primary only 또는 Primary + Wrist
- proprio disabled
- `window_size=2`
- `action_horizon=4`
- `action_dim=7`
- batch size 4
- seed 42
- Adam 계열 Octo optimizer 설정, peak LR `3e-4`
- warmup 500 steps, cosine decay
- weight decay `0.01`, gradient clipping `1.0`
- evaluation 500-step 간격
- W&B offline mode

### 10.1 Preflight와 smoke test

```bash
octo-train \
  --dataset-dir data/pick_place/rlds/ur5e_pick_place_5ep_1val/ur5e_pick/1.0.0
```

위 명령은 dataset, GPU, Octo checkout, Conda 환경과 실제 실행 명령을 출력할 뿐 학습하지 않습니다.

10-step smoke test:

```bash
octo-train \
  --dataset-dir data/pick_place/rlds/ur5e_pick_place_5ep_1val/ur5e_pick/1.0.0 \
  --execute --smoke
```

### 10.2 Primary-only 5K

`octo-train --execute`는 기본 Primary-only 5K 설정을 실행합니다.

```bash
octo-train \
  --dataset-dir data/pick_place/rlds/ur5e_pick_place_5ep_1val/ur5e_pick/1.0.0 \
  --execute
```

### 10.3 Primary + Wrist 5K

Wrist 모델은 config suffix `wrist`가 필요합니다. `image_obs_keys.wrist=None`을 CLI로 덮어쓰지 말고 config variant를 사용하십시오.

```bash
cd /home/sixr/octo_ur5e_finetuning
conda activate octo_env

export PYTHONPATH=/home/sixr/octo_ur5e_finetuning
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export WANDB_DIR=/home/sixr/octo_ur5e_finetuning/runs/wandb

python /home/sixr/Desktop/octo/scripts/finetune.py \
  --config=/home/sixr/octo_ur5e_finetuning/training/finetune_config.py:head_only,language_conditioned,wrist \
  --name=pick_place_5ep_primary_wrist_5k \
  --config.dataset_kwargs.data_dir=/home/sixr/octo_ur5e_finetuning/data/pick_place/rlds/ur5e_pick_place_5ep_1val \
  --config.save_dir=/home/sixr/octo_ur5e_finetuning/runs \
  --config.num_steps=5000 \
  --config.eval_interval=500 \
  --config.save_interval=1000
```

학습 시작 로그에서 Wrist가 zero padding으로 대체되지 않고 실제 `image_wrist`가 observation tokenizer에 연결되는지 확인하십시오. RLDS 검사 결과 `wrist_valid`도 모두 true여야 합니다.

### 10.4 백그라운드 실행

예시는 `nohup`이며 동일 run name으로 중복 실행하지 마십시오.

```bash
mkdir -p runs/logs

nohup bash -lc '
  cd /home/sixr/octo_ur5e_finetuning
  source /home/sixr/miniconda3/etc/profile.d/conda.sh
  conda activate octo_env
  export PYTHONPATH=/home/sixr/octo_ur5e_finetuning
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export WANDB_MODE=offline
  export WANDB_DIR=/home/sixr/octo_ur5e_finetuning/runs/wandb
  python /home/sixr/Desktop/octo/scripts/finetune.py \
    --config=/home/sixr/octo_ur5e_finetuning/training/finetune_config.py:head_only,language_conditioned,wrist \
    --name=pick_place_5ep_primary_wrist_5k \
    --config.dataset_kwargs.data_dir=/home/sixr/octo_ur5e_finetuning/data/pick_place/rlds/ur5e_pick_place_5ep_1val \
    --config.save_dir=/home/sixr/octo_ur5e_finetuning/runs \
    --config.num_steps=5000 \
    --config.eval_interval=500 \
    --config.save_interval=1000
' > runs/logs/pick_place_5ep_primary_wrist_5k.log 2>&1 &
```

```bash
tail -f runs/logs/pick_place_5ep_primary_wrist_5k.log
```

### 10.5 결과와 loss graph

```bash
octo-plot-metrics runs/octo_ur5e/pick/<run_directory>
```

생성 파일:

- `loss.png`: train raw/moving-average와 validation loss
- `metrics.csv`: step별 loss
- `metrics_summary.json`: final/best 판단에 필요한 요약

검증 episode 수가 적으면 validation loss 변동이 큽니다. 마지막 checkpoint만 보지 말고 최저 validation step과 실제 로봇 동작을 함께 비교하십시오.

### 10.6 Resume

공식 fine-tuner가 full TrainState resume flag를 제공하지 않는 현재 구성에서는 `training/resume_finetune.py`가 upstream 파일을 수정하지 않고 메모리에서 검증된 작은 patch를 적용합니다. resume에는 모델 checkpoint가 아니라 `state/<step>/default`의 optimizer/RNG 포함 full TrainState가 필요합니다.

```bash
export OCTO_RESUME_STATE=/absolute/path/to/run/state/5000
export OCTO_FINETUNE_SCRIPT=/home/sixr/Desktop/octo/scripts/finetune.py

python training/resume_finetune.py \
  --config=training/finetune_config.py:head_only,language_conditioned,wrist \
  --name=<new_run_name> \
  --config.dataset_kwargs.data_dir=<rlds_parent> \
  --config.save_dir=/home/sixr/octo_ur5e_finetuning/runs \
  --config.num_steps=10000
```

항상 새 run name을 사용하여 기존 checkpoint와 결과를 덮어쓰지 마십시오.

## 11. Inference

`octo-policy`는 기본적으로 dry-run입니다. 두 영상 frame을 학습과 같은 0.1초 간격으로 ring buffer에서 선택하고, 모델의 4-step action chunk를 각 0.1초 trajectory point로 실행합니다.

실행 전 필요 조건:

- 학습에 사용한 instruction과 동일한 문장
- 해당 모델이 Primary-only인지 Primary+Wrist인지 확인
- UR driver와 PolyScope External Control 실행
- `scaled_joint_trajectory_controller` active
- Primary/Wrist topic 정상
- MoveIt `/compute_ik` service 실행 (`--execute` 시)
- checkpoint directory와 step 확인

### 11.1 Dry-run

Primary + Wrist Pick & Place 예시:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate

octo-policy \
  --checkpoint runs/octo_ur5e/pick/<run_directory> \
  --step 5000 \
  --config collector/config/pick_place.yaml \
  --instruction "pick up the blue object and place it in the tray" \
  --use-wrist \
  --max-steps 20
```

### 11.2 안전한 최초 실제 실행

먼저 한 action만 1 mm 제한으로 좌표계와 방향을 검증합니다.

```bash
octo-policy \
  --checkpoint runs/octo_ur5e/pick/<run_directory> \
  --step 5000 \
  --config collector/config/pick_place.yaml \
  --instruction "pick up the blue object and place it in the tray" \
  --use-wrist \
  --max-steps 1 \
  --action-chunk-steps 1 \
  --max-translation-m 0.001 \
  --max-rotation-rad 0.005 \
  --command-duration-sec 0.1 \
  --execute --confirm-real-robot
```

방향과 좌표계가 맞고 목표 pose에 실제로 도달하는 것을 확인한 뒤 translation clamp를 `0.005`, 이후 `0.010` m로 단계적으로 높이고 `max-steps`를 늘리십시오.

기본 gripper 안정화 설정:

- close threshold: `0.9`
- open threshold: `0.1`
- close debounce: 3 step
- open debounce: 3 step
- 두 threshold 사이: 이전 상태 유지

필요하면 CLI의 `--gripper-*-threshold`, `--gripper-*-debounce-steps`로 조정할 수 있습니다.

실행 JSONL은 기본적으로 `runs/policy_logs/`에 저장되며 다음을 포함합니다.

- `run_start`, 각 `step`, `run_stop`
- `current_step`, `max_steps`, `stop_reason`
- raw `policy_action`
- `clamped_command`
- command/actual TCP translation norm과 비율
- target pose error와 도달 여부
- controller timing 및 이전 goal 완료 여부
- gripper policy value, semantic state, hysteresis/debounce 설정

정책이 멈춘 경우 마지막 `run_stop.stop_reason`으로 `max_steps_reached`, keyboard interrupt, ROS shutdown, safety/IK 오류를 먼저 구분하십시오.

## 12. 주요 CLI 요약

| 명령 | 용도 |
|---|---|
| `octo-collector doctor` | ROS/config/hardware read-only preflight |
| `octo-collector record-demo` | 프리드라이브 시연 기록 |
| `octo-collector validate-demo` | demonstration trajectory 검증 |
| `octo-collector inspect` | episode metadata 조회 |
| `octo-collector replay` | 궤적 dry-run 또는 실제 replay/capture |
| `octo-wrist-camera` | oCam GRBG ROS Image publisher |
| `octo-process-episode` | episode 하나를 10 Hz transition으로 처리 |
| `octo-process-batch` | raw root 전체 처리 |
| `octo-inspect-processed` | processed NPZ shape/statistics 확인 |
| `octo-evaluate-processed` | 처리 품질 재평가 |
| `octo-dataset doctor` | processing/RLDS dependency와 상태 검사 |
| `octo-dataset process` | pending/stale episode 증분 처리 |
| `octo-dataset plan` | 품질 필터와 episode-level split 미리보기 |
| `octo-dataset build` | Processing + RLDS/TFDS 생성 |
| `octo-check-dataset` | TFDS split/schema/shape 검사 |
| `octo-train` | 학습 preflight, smoke, 기본 Primary 학습 |
| `octo-plot-metrics` | W&B offline history를 CSV/PNG로 변환 |
| `octo-policy` | Octo dry-run 또는 guarded robot inference |

각 명령의 현재 옵션은 구현이 기준입니다.

```bash
octo-collector record-demo --help
octo-collector replay --help
octo-dataset build --help
octo-policy --help
```

## 13. 권장 전체 작업 순서

```text
1. UR driver / External Control / cameras / MoveIt 시작
2. octo-collector doctor로 topic, controller, safety, IO 검사
3. record-demo로 동일 instruction의 시연 수집
4. validate-demo와 inspect로 raw demonstration 확인
5. replay dry-run
6. replay --execute로 양 카메라와 실제 상태 기록
7. quality_report와 영상 확인
8. octo-dataset plan으로 포함 episode와 train/val split 확인
9. octo-dataset build --execute로 Processing + RLDS 생성
10. octo-check-dataset으로 image/action/mask/split 검사
11. octo-train smoke test
12. Primary-only 또는 Primary+Wrist fine-tuning
13. octo-plot-metrics로 train/validation curve 확인
14. octo-policy dry-run
15. 1-step, 1 mm 실제 안전 검증
16. clamp와 max_steps를 단계적으로 확장
```

## 14. 문제 해결

### `robot_program_running`이 false이거나 controller가 inactive

PolyScope External Control 프로그램을 다시 Play하고 확인합니다.

```bash
ros2 control list_controllers
ros2 topic echo /io_and_status_controller/robot_program_running --once
```

`scaled_joint_trajectory_controller`가 inactive이면 move-to-start, replay, inference action goal이 거절됩니다.

### Wrist camera를 열 수 없음

- `/dev/v4l/by-id/...`가 `/dev/video*`로 연결되는지 확인
- 현재 사용자가 `video` group/ACL 권한을 갖는지 확인
- `fuser`로 다른 프로세스가 장치를 점유하는지 확인
- 일반 `v4l2_camera` 대신 프로젝트의 `octo-wrist-camera` 사용

oCam은 GRBG만 제공하므로 encoding이 빈 문자열인 일반 변환 경로에서는 OpenCV demosaicing 오류가 날 수 있습니다.

### Replay가 계획 시간보다 오래 걸림

controller feedback의 progress와 speed scaling을 확인하십시오. planned duration은 controller trajectory time이며 wall-clock duration과 다를 수 있습니다. 완료는 action result `error_code=0`으로 판단합니다.

### Wrist fine-tuning에서 zero padding 메시지

RLDS에 `image_wrist`/`wrist_valid`가 있는지, `octo-check-dataset` 결과가 정상인지, config가 정확히 `head_only,language_conditioned,wrist`인지 확인하십시오. `None` field를 CLI override하려 하면 `ml_collections` flag parsing error가 발생합니다.

### `ModuleNotFoundError: flax`

ROS `.venv`의 Python으로 공식 Octo fine-tuner를 실행한 것입니다. `conda activate octo_env` 후 `which python`과 Octo import를 확인하십시오.

### CUDA factory 중복 경고 / TensorRT 경고

학습이 계속 진행되고 JAX GPU device가 정상이라면 보통 비치명적 초기화 경고입니다. 실제 종료 여부는 traceback, process status, checkpoint 생성으로 판단하십시오.

## 15. 추가 문서

- `docs/collector_spec.md`: collector/replay 안전 및 저장 세부 규격
- `processing/README.md`: timestamp, segment, action 처리 규약
- `rlds_builder/README.md`: RLDS builder와 schema 요약
- `docs/official_sources.md`: 공식 참고 자료

새로운 task를 추가할 때는 기존 기준 데이터의 config와 raw/RLDS를 복사·수정하지 말고, 새 collector YAML, 새 `storage.output_root`, 새 RLDS output, 새 training run name을 사용하십시오.
