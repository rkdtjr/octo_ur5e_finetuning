# Codex 구현 지시 — UR5e Demonstration Collector v2

- Status: active
- Target branch: `feature/collector`
- Prompt path: `codex_prompts/collector_v2.md`
- Repository: `octo_ur5e_finetuning`
- ROS distribution: ROS 2 Jazzy
- Robot: Universal Robots UR5e
- Gripper command: Standard Digital Output 1
- Dataset gripper semantic: `0=open`, `1=closed`
- Created: 2026-07-30

---

## 1. 역할과 목표

당신은 ROS 2 Jazzy, Universal Robots ROS 2 Driver, ros2_control, 로봇 demonstration 수집, SE(3) 좌표변환, RLDS 및 Octo 파인튜닝 파이프라인에 익숙한 시니어 로보틱스 소프트웨어 엔지니어다.

현재 저장소에는 프로젝트 스캐폴드만 있고 collector 관련 Python 파일은 대부분 placeholder다. 이번 작업은 `feature/collector` 브랜치에서 **실제로 사용할 수 있는 UR5e demonstration collector와 trajectory replay 기능을 구현하는 것**이다.

최종 파이프라인은 다음 순서를 따른다.

```text
1. 사용자가 UR5e를 freedrive로 직접 움직인다.
2. 키보드로 그리퍼를 열고 닫는다.
3. joint/TCP/그리퍼 명령과 원시 ROS 데이터를 보존한다.
4. 기록된 joint trajectory를 검증한다.
5. trajectory를 UR5e에서 재생한다.
6. 재생 중 primary/wrist image와 actual robot state를 rosbag2로 기록한다.
7. 후속 브랜치에서 동기화, 품질 평가, RLDS 변환, Octo fine-tuning을 수행한다.
```

이번 브랜치에서는 1~6을 구현한다. 후속 단계가 collector 데이터 계약을 그대로 사용할 수 있도록 설계하되, RLDS builder나 Octo 학습 코드는 구현하지 않는다.

---

## 2. 가장 중요한 확정 사항

### 2.1 브랜치

반드시 다음 브랜치에서만 작업한다.

```text
feature/collector
```

`main`에 직접 구현하거나 merge하지 않는다.

### 2.2 그리퍼 제어

기존 `rg_grip()` URScript 방식은 사용하지 않는다.

그리퍼는 Universal Robots의 **Standard Digital Output 1**로 제어한다.

```text
ROS service: /io_and_status_controller/set_io
Service type: ur_msgs/srv/SetIO
fun: set digital output
pin: 1
state: 0.0 or 1.0
```

데이터셋 semantic은 항상 다음과 같다.

```text
0 = open
1 = closed
```

물리 출력 polarity는 설정으로 분리한다.

```text
output_value_for_open
output_value_for_closed
```

현재 기본값은 다음과 같다.

```text
open  -> DO1 = 0
close -> DO1 = 1
```

단, 이 polarity는 실제 배선과 장비 설정에 따라 바뀔 수 있으므로 코드에 숨겨서 하드코딩하지 않는다.

### 2.3 키보드 동작

`record-demo --execute` 실행 중에는 키보드 입력이 실제 gripper command로 이어져야 한다.

기본 키 매핑:

```text
0 또는 o: gripper open  -> semantic 0
1 또는 c: gripper close -> semantic 1
q: 기록 정상 종료 및 저장
Esc: 기록 중단 및 aborted 처리
```

`--execute`가 없으면 키 입력과 event는 기록할 수 있지만 실제 `SetIO` service는 호출하지 않는 dry-run이어야 한다.

키보드 입력 처리는 ROS executor를 block하면 안 된다. 터미널 입력 thread 또는 비동기 reader가 command queue에 event를 넣고, ROS node가 queue를 소비해 service를 호출하도록 분리한다.

### 2.4 Freedrive

Collector의 목적은 사용자가 freedrive 상태에서 로봇을 손으로 움직이면서 demonstration을 기록하는 것이다.

그러나 이 브랜치에서는 다음을 자동화하지 않는다.

- robot power on
- brake release
- PolyScope program start
- freedrive enable/disable
- controller 자동 전환

사용자가 freedrive를 활성화한 상태에서 collector를 실행하는 것을 기본으로 한다. 사용 중인 freedrive 방식이 teach pendant인지 `freedrive_mode_controller`인지 추측하지 않는다.

실제 gripper command를 전송하려면 UR ROS 2 Driver와 I/O service가 사용 가능한 상태여야 한다. `--execute` 실행 전 doctor/preflight에서 service availability, program state, safety state를 검사한다.

### 2.5 Raw 우선

수집 중 데이터셋을 바로 10 Hz로 확정하지 않는다.

- 원시 ROS topic은 rosbag2로 원래 timestamp와 함께 보존한다.
- demonstration replay용 structured trajectory는 별도 파일로 만든다.
- image/state 동기화와 10 Hz dataset 생성은 후속 `feature/synchronization` 브랜치에서 수행한다.
- 원시 image를 임의로 JPEG/PNG로 다시 인코딩하지 않는다.

### 2.6 좌표변환

UR TCP pose는 다음 표현을 사용한다.

```text
[x, y, z, rx, ry, rz]
```

- `x, y, z`: meter
- `rx, ry, rz`: Euler angle이 아니라 rotation vector

pure-Python core에서 다음을 구현하고 테스트한다.

```text
UR pose6 <-> 4x4 homogeneous transform
quaternion pose <-> 4x4 homogeneous transform
tool-frame relative delta
base-frame relative delta
relative delta 적용 및 원 pose 복원
```

---

## 3. 작업 시작 전에 반드시 할 일

### 3.1 저장소 검사

먼저 다음을 실행해 현재 상태를 확인한다.

```bash
git status --short
git branch --show-current
git log --oneline --decorate -10
find . -maxdepth 4 -type f | sort
```

현재 스캐폴드에는 최소한 다음 파일이 있다.

```text
README.md
pyproject.toml
collector/config/collector.yaml
collector/octo_ur5e_collector/collector_cli.py
collector/octo_ur5e_collector/record_demonstration_node.py
collector/octo_ur5e_collector/replay_trajectory_node.py
collector/octo_ur5e_collector/record_replay_dataset_node.py
processing/build_actions.py
processing/synchronize_episode.py
processing/evaluate_quality.py
processing/convert_to_rlds.py
rlds_builder/ur5e_pick/ur5e_pick_dataset_builder.py
docs/collector_spec.md
docs/official_sources.md
tests/
```

기존 사용자 변경사항을 삭제하거나 reset하지 않는다.

### 3.2 브랜치 처리

현재 브랜치가 `feature/collector`가 아니면 다음 원칙을 따른다.

1. working tree가 깨끗하면 `feature/collector`로 switch한다.
2. 브랜치가 없으면 현재 기준점에서 생성한다.
3. working tree가 dirty라서 안전하게 switch할 수 없으면 reset/stash하지 말고 작업을 중단한 뒤 상태를 보고한다.

예시:

```bash
git switch feature/collector
# 또는
git switch -c feature/collector
```

### 3.3 구현 전 보고

코드를 수정하기 전에 다음을 20줄 이내로 출력한다.

- 확인한 현재 구조
- 변경할 파일
- 새로 만들 파일
- 데이터 흐름
- 실제 하드웨어에서 검증이 필요한 항목

---

## 4. 반드시 확인할 공식 자료

아래 자료를 먼저 직접 열어 현재 내용을 확인한다. 링크만 문서에 복사하고 내용을 보지 않은 채 구현하지 않는다.

### 4.1 Octo 공식 자료

#### 공식 저장소

https://github.com/octo-models/octo

확인할 내용:

- 공식 checkpoint 로딩 방식
- `octo-small-1.5` 명칭
- 공식 fine-tuning 진입점
- observation/action dictionary 구조
- RLDS dataset loading 흐름

#### 새 observation/action 공간 최소 fine-tuning 예제

https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.py

이 예제는 simulated ALOHA용이다. 다음 값은 UR5e에 그대로 복사하지 않는다.

```text
action_horizon = 50
action_dim = 14
L1ActionHead
```

UR5e 프로젝트의 action 후보는 다음 7차원이다.

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

Collector에서는 action을 최종 생성하지 않지만, 저장되는 pose/gripper 데이터가 후속 action builder에서 이 형식으로 변환 가능해야 한다.

#### 공식 advanced fine-tuning script

https://github.com/octo-models/octo/blob/main/scripts/finetune.py

#### 공식 fine-tuning config

https://github.com/octo-models/octo/blob/main/scripts/configs/finetune_config.py

확인할 내용:

- `image_obs_keys`
- `proprio_obs_key`
- `language_key`
- `standardize_fn`
- `action_normalization_mask`
- `window_size`
- `action_horizon`
- `full`, `head_only`, `head_mlp_only`
- `image_conditioned`, `language_conditioned`, `multimodal`

공식 config에는 7차원 action의 마지막 gripper 차원을 normalization에서 제외하는 예제가 있다.

```python
[True, True, True, True, True, True, False]
```

이 값은 action dimension이 7일 때만 유효하므로 config validation에서 길이를 검사한다.

#### 공식 dataloader 예제와 구현

https://github.com/octo-models/octo/blob/main/examples/05_dataloading.ipynb

https://github.com/octo-models/octo/blob/main/octo/data/dataset.py

#### 공식 fine-tuned evaluation 예제

https://github.com/octo-models/octo/blob/main/examples/03_eval_finetuned.py

### 4.2 Universal Robots ROS 2 공식 자료

#### UR ROS 2 Driver Jazzy 문서

https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/jazzy/index.html

#### UR controller 문서

https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/jazzy/doc/ur_robot_driver/ur_controllers/doc/index.html

확인할 내용:

- `io_and_status_controller`
- `~/io_states [ur_msgs/msg/IOStates]`
- `~/set_io [ur_msgs/srv/SetIO]`
- `~/robot_program_running`
- `~/safety_mode`
- scaled trajectory controller
- speed scaling에 따른 trajectory 진행

#### UR ROS 2 Driver Jazzy source

https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/tree/jazzy

#### UR pose 표현

https://www.universal-robots.com/articles/ur/programming/read-a-single-coordinate-or-axis-rotation/

https://www.universal-robots.com/articles/ur/programming/axis-angle-representation/

### 4.3 좌표변환 라이브러리

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html

직접 만든 불완전한 Rodrigues 근사 대신 `scipy.spatial.transform.Rotation`을 사용한다.

### 4.4 공식 자료 버전 기록

`docs/official_sources.md`를 실제 기록 문서로 바꾼다.

각 source마다 다음을 기록한다.

```text
source_name
url
checked_at_utc
branch_or_commit
installed_version_if_applicable
used_for
project_specific_difference
```

가능하면 구현 시점의 Git commit SHA를 기록한다. `main` 링크만 기록하고 끝내지 않는다.

공식 자료와 설치된 ROS interface가 다르면 설치된 interface를 다음 명령으로 확인하고 차이를 문서화한다.

```bash
ros2 interface show ur_msgs/srv/SetIO
ros2 interface show ur_msgs/msg/IOStates
ros2 interface show control_msgs/action/FollowJointTrajectory
ros2 topic list
ros2 service list
ros2 action list
```

---

## 5. 이번 브랜치의 범위

### 구현한다

1. collector Python package와 console CLI
2. typed YAML config와 validation
3. episode directory, manifest, status 관리
4. collector doctor/preflight
5. freedrive demonstration recording
6. 키보드 기반 Digital Output 1 gripper 제어
7. demonstration raw rosbag2 기록
8. structured replay trajectory 생성
9. trajectory validation
10. dry-run replay
11. 명시적 `--execute` actual replay
12. replay 중 raw camera/state rosbag2 기록
13. gripper event replay
14. UR pose/SE(3) transformation utility
15. pure-Python unit tests
16. collector 사용 문서

### 구현하지 않는다

- 최종 image/state synchronization
- 최종 quality pass/fail 판정 시스템
- RLDS builder 완성
- Octo fine-tuning 실행
- inference node
- Docker GPU training 환경 수정
- hand-eye calibration
- object detection
- automatic task segmentation
- robot power/brake/program 자동 제어
- freedrive controller 자동 전환

다음 디렉터리는 collector 구현에 꼭 필요한 공용 interface를 제외하고 수정하지 않는다.

```text
processing/
rlds_builder/
training/
inference/
docker/
```

---

## 6. 권장 패키지 구조

기존 파일을 최대한 살리되 pure core와 ROS adapter를 분리한다.

```text
collector/octo_ur5e_collector/
├── __init__.py
├── collector_cli.py
├── record_demonstration_node.py
├── replay_trajectory_node.py
├── record_replay_dataset_node.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── episode.py
│   ├── schema.py
│   ├── trajectory.py
│   ├── transforms.py
│   ├── replay_scheduler.py
│   ├── keyboard_commands.py
│   └── time_utils.py
└── ros_adapters/
    ├── __init__.py
    ├── state_source.py
    ├── tf_source.py
    ├── digital_output_gripper.py
    ├── trajectory_client.py
    ├── rosbag_recorder.py
    └── preflight.py
```

이름은 기존 구조에 맞게 조정할 수 있지만 다음 원칙은 유지한다.

- `core/`는 ROS가 없는 Python 환경에서도 import 가능
- `ros_adapters/`만 `rclpy`, ROS message/service/action을 import
- ROS package를 PyPI dependency로 선언하지 않음
- hardware access는 adapter interface 뒤에 숨김
- unit test는 fake adapter로 실행 가능

---

## 7. 패키징과 CLI

루트 `pyproject.toml`을 보완해 editable install과 console script를 지원한다.

```toml
[project.scripts]
octo-collector = "octo_ur5e_collector.collector_cli:main"

[tool.setuptools.packages.find]
where = ["collector"]
```

최소 CLI:

```text
octo-collector doctor
octo-collector record-demo
octo-collector validate-demo
octo-collector replay
octo-collector inspect
```

도움말 예시:

```bash
octo-collector --help
octo-collector record-demo --help
octo-collector replay --help
```

### 7.1 doctor

```bash
octo-collector doctor --config collector/config/collector.yaml
```

다음을 검사한다.

- config validation
- ROS graph 연결
- joint state topic 존재
- TF source 존재
- IO state topic 존재
- SetIO service 존재
- FollowJointTrajectory action 존재
- camera topic 존재 여부
- robot program state
- safety state
- configured frames
- rosbag storage plugin
- output directory write permission

`doctor`는 절대로 로봇을 움직이거나 output을 변경하지 않는다.

### 7.2 record-demo

```bash
octo-collector record-demo \
  --config collector/config/collector.yaml \
  --instruction "pick up the blue object" \
  --execute
```

- `--execute`: 키보드 gripper command를 실제 DO1로 전송
- `--execute` 없음: gripper command dry-run
- robot arm trajectory command는 record-demo에서 절대 전송하지 않음
- 사용자가 freedrive로 직접 움직임

### 7.3 validate-demo

```bash
octo-collector validate-demo data/raw/<episode_id>
```

### 7.4 replay

```bash
# 기본은 dry-run
octo-collector replay data/raw/<episode_id>

# 실제 UR5e 실행
octo-collector replay data/raw/<episode_id> --execute
```

`--execute` 없이는 trajectory goal과 SetIO를 실제로 보내면 안 된다.

### 7.5 inspect

```bash
octo-collector inspect data/raw/<episode_id>
```

manifest, 상태, sample 수, duration, gripper transition, topic 존재 여부, validation 결과를 사람이 읽기 좋은 형태로 출력한다.

---

## 8. Config schema

기존 `collector/config/collector.yaml`을 다음 요구사항을 만족하도록 확장한다. 실제 구현에서는 dataclass 또는 명확한 typed model을 사용한다.

```yaml
schema_version: 2

robot:
  name: ur5e
  base_frame: base
  tcp_frame: tool0
  joint_names:
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint

ros:
  joint_state_topic: /joint_states
  tf_topic: /tf
  tf_static_topic: /tf_static
  controller_state_topic: /scaled_joint_trajectory_controller/controller_state
  trajectory_action: /scaled_joint_trajectory_controller/follow_joint_trajectory
  io_states_topic: /io_and_status_controller/io_states
  set_io_service: /io_and_status_controller/set_io
  robot_program_running_topic: /io_and_status_controller/robot_program_running
  safety_mode_topic: /io_and_status_controller/safety_mode

cameras:
  - logical_name: primary
    image_topic: /camera/camera/color/image_raw
    camera_info_topic: /camera/camera/color/camera_info
    required: true
  - logical_name: wrist
    image_topic: /wrist_camera/image_raw
    camera_info_topic: null
    required: true

sampling:
  demonstration_rate_hz: 100.0
  target_dataset_rate_hz: 10.0

freedrive:
  activation: external
  auto_enable: false
  auto_disable: false

keyboard:
  open_keys: ["0", "o"]
  close_keys: ["1", "c"]
  finish_keys: ["q"]
  abort_keys: ["esc"]

gripper:
  semantic_open: 0
  semantic_closed: 1
  backend: ur_standard_digital_output
  output_pin: 1
  output_value_for_open: 0.0
  output_value_for_closed: 1.0
  command_on_change_only: true
  minimum_command_interval_sec: 0.2
  command_timeout_sec: 2.0
  confirmation_timeout_sec: 1.0
  readback_from_io_states: true
  initial_state_source: io_readback

replay:
  controller_joint_order:
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint
  initial_joint_tolerance_rad: 0.05
  speed_scale: 1.0
  start_settle_sec: 1.0
  end_settle_sec: 1.0
  feedback_stale_sec: 0.5
  max_joint_velocity_rad_s: null
  max_joint_acceleration_rad_s2: null
  execute_requires_program_running: true
  execute_requires_normal_safety: true

storage:
  output_root: data/raw
  rosbag_storage_id: mcap
  overwrite: false

raw_topics:
  demonstration:
    - /joint_states
    - /tf
    - /tf_static
    - /io_and_status_controller/io_states
    - /io_and_status_controller/robot_program_running
    - /io_and_status_controller/safety_mode
  replay:
    - /joint_states
    - /tf
    - /tf_static
    - /scaled_joint_trajectory_controller/controller_state
    - /io_and_status_controller/io_states
    - /io_and_status_controller/robot_program_running
    - /io_and_status_controller/safety_mode
    - /camera/camera/color/image_raw
    - /camera/camera/color/camera_info
    - /wrist_camera/image_raw

action_contract:
  dimension: 7
  translation_unit: meter
  rotation_representation: rotation_vector
  delta_frame: tool
  gripper_index: 6
  normalization_mask: [true, true, true, true, true, true, false]
```

주의: 위 YAML 예시의 `gripper:` 앞 불필요한 공백은 실제 파일에서 제거한다.

필수 validation:

- joint name은 정확히 6개이며 중복 없음
- semantic open/closed는 각각 0/1이고 서로 다름
- physical output open/closed 값은 서로 다름
- output pin은 유효한 정수
- rate는 양수
- timeout은 음수가 아님
- action dimension과 normalization mask 길이가 같음
- gripper index가 action dimension 안에 있음
- camera logical name 중복 없음
- output directory overwrite 정책 확인
- unknown config key를 조용히 무시하지 않음

---

## 9. Episode 저장 형식

각 episode는 덮어쓰지 않는 고유 디렉터리로 저장한다.

```text
data/raw/<episode_id>/
├── manifest.json
├── status.json
├── config_resolved.yaml
├── demonstration/
│   ├── rosbag2/
│   ├── events.jsonl
│   ├── samples.npz
│   ├── trajectory.npz
│   ├── trajectory_metadata.json
│   └── validation.json
└── replay/
    ├── rosbag2/
    ├── events.jsonl
    ├── execution_summary.json
    └── validation.json
```

권장 episode id:

```text
20260730_180000_<short_uuid>
```

### 9.1 manifest.json

최소 필드:

```json
{
  "schema_version": 2,
  "episode_id": "...",
  "instruction": "pick up the blue object",
  "robot": "ur5e",
  "base_frame": "base",
  "tcp_frame": "tool0",
  "joint_names": [],
  "camera_topics": {},
  "gripper_semantic": {"open": 0, "closed": 1},
  "gripper_output": {"pin": 1, "open": 0.0, "closed": 1.0},
  "transform_convention": "T_A_B maps p_B to p_A",
  "created_at_utc": "...",
  "git": {"branch": "feature/collector", "commit": "...", "dirty": false},
  "collector_version": "...",
  "hardware_verification": []
}
```

### 9.2 status.json

상태 전이를 명시한다.

```text
created
recording_demo
demo_recorded
demo_validated
replaying
completed
aborted
failed
```

status update는 임시 파일에 쓴 후 atomic rename한다. 중간 실패 시 episode를 삭제하지 않는다.

### 9.3 events.jsonl

모든 event는 다음 시간을 가능하면 함께 가진다.

```text
source_time_ns
receipt_time_ns
monotonic_time_ns
elapsed_sec
```

이벤트 예:

```json
{
  "event": "keyboard_gripper_command",
  "key": "1",
  "semantic_state": 1,
  "output_pin": 1,
  "output_value": 1.0,
  "execute": true,
  "service_called": true,
  "service_success": true,
  "readback_confirmed": true,
  "elapsed_sec": 2.417
}
```

---

## 10. Demonstration recorder

### 10.1 수집 동작

`record-demo` 시작 시:

1. config 로드 및 validation
2. episode 생성
3. preflight 실행
4. demonstration rosbag recorder 시작
5. state subscription 준비 확인
6. keyboard listener 시작
7. configured rate로 structured sample 기록
8. 사용자 freedrive 조작
9. 키보드 gripper command 처리
10. `q`에서 정상 종료
11. raw bag 종료
12. structured trajectory finalize
13. validation 실행
14. manifest/status 저장

### 10.2 structured sample

최소 필드:

```text
elapsed_sec
monotonic_time_ns
joint_source_time_ns
joint_receipt_time_ns
joint_position[6]
joint_velocity[6] or NaN
tcp_source_time_ns
tcp_receipt_time_ns
tcp_pose6[6]
tcp_matrix[4,4] or reconstructable pose6
gripper_semantic_state
digital_output_value
digital_output_source_time_ns
state_age_ms
tf_age_ms
```

매 sample마다 Python object JSON을 반복 저장해 성능을 떨어뜨리지 말고 메모리 buffer 또는 chunked writer를 사용한 뒤 `npz`로 finalize한다. 비정상 종료에도 가능한 범위에서 임시 데이터를 복구할 수 있어야 한다.

### 10.3 Joint ordering

수신된 `JointState.name` 순서가 항상 controller 순서와 같다고 가정하지 않는다.

- config의 joint name 순서로 재정렬
- 누락 joint가 있으면 sample invalid
- duplicate joint name은 오류
- NaN/Inf는 오류

### 10.4 TCP pose

TCP는 `base_frame -> tcp_frame` transform으로 기록한다.

- TF가 없거나 stale이면 sample invalid 또는 recorder failure 정책을 명확히 적용
- calibration mismatch 경고가 있는 환경에서 TCP 정확도를 보장한다고 주장하지 않음
- source/receipt time과 age를 기록

### 10.5 Raw rosbag

Demonstration에서도 raw rosbag을 기록한다. 최소 topic은 config의 `raw_topics.demonstration`을 따른다.

`RosbagRecorder`는 다음 중 하나로 구현할 수 있다.

1. `rosbag2_py`
2. 안전하게 관리되는 `ros2 bag record` subprocess

subprocess를 사용할 경우:

- shell string이 아니라 argument list 사용
- PID 보존
- 정상 종료는 SIGINT
- timeout 후 강제 종료 처리
- stderr/stdout log 저장
- metadata 존재 확인
- 종료 코드를 manifest에 기록

### 10.6 초기 gripper 상태

기록 시작 시 configured DO1 readback을 읽어 semantic state로 변환한다.

- open/closed polarity와 일치하면 semantic state 확정
- readback이 없으면 `--initial-gripper {open,closed}` 또는 명확한 config가 필요
- 모르는 상태를 임의로 open으로 가정하지 않음

---

## 11. Digital Output gripper adapter

`DigitalOutputGripper`를 명확한 adapter로 구현한다.

필수 API 예시:

```python
class DigitalOutputGripper:
    def command_semantic(self, state: int, *, execute: bool) -> GripperCommandResult:
        ...

    def latest_readback(self) -> DigitalOutputReadback | None:
        ...

    def semantic_from_output(self, value: float) -> int:
        ...

    def output_from_semantic(self, state: int) -> float:
        ...
```

### 11.1 SetIO request

설치된 `ur_msgs/srv/SetIO` interface를 먼저 확인한다.

```bash
ros2 interface show ur_msgs/srv/SetIO
```

가능하면 message에 정의된 상수를 사용한다. 상수가 Python에서 제공되지 않으면 공식 interface에서 확인한 값을 한 곳에 명시하고 테스트한다.

기본 의미:

```text
fun = 1
pin = configured output pin, default 1
state = configured 0.0 or 1.0
```

### 11.2 command 처리

- semantic state는 0 또는 1만 허용
- 동일 semantic state 반복 전송 방지
- minimum interval 적용
- service wait timeout
- async response timeout
- response success 확인
- IOStates readback confirmation
- command와 confirmation latency 기록
- dry-run은 service를 절대 호출하지 않음

### 11.3 실패 처리

record-demo actual command 실패 시:

- 실패 event 저장
- 사용자에게 즉시 출력
- episode를 `failed` 또는 정책에 따라 `aborted`로 종료
- 실패를 무시하고 잘못된 gripper label로 계속 기록하지 않음

replay command 실패 시:

- trajectory goal cancel
- rosbag 안전 종료
- execution summary에 원인 기록
- episode raw data 유지

### 11.4 의미 제한

`IOStates` readback은 controller의 output 상태다. 다음을 의미하지 않는다.

- 실제 finger width
- object detected
- grasp success
- mechanical completion

코드와 문서에서 `actual_gripper_state`라는 오해 가능한 이름을 쓰지 않는다.

---

## 12. Structured trajectory 생성과 validation

Demonstration sample에서 replay용 trajectory를 생성한다.

필수 배열:

```text
time_from_start_sec: shape (N,)
joint_position: shape (N,6)
gripper_semantic_state: shape (N,)
tcp_pose6: shape (N,6)
```

선택 배열:

```text
joint_velocity
digital_output_value
source timestamps
validity masks
```

### 12.1 시간

- 첫 valid sample을 `t=0`으로 이동
- strictly increasing
- duplicate/non-monotonic timestamp 처리 정책 명시
- duration > 0
- configured replay speed scale 반영 가능

### 12.2 그리퍼

- zero-order hold
- transition index와 transition time을 별도로 추출
- 같은 state 반복 event 제거
- 첫 state를 명시

예:

```text
semantic samples: 0 0 0 1 1 1 0 0
replay events: close at t3, open at t6
```

### 12.3 validation

최소 검사:

- sample 수
- duration
- shape
- finite values
- joint ordering
- timestamp monotonicity
- max joint step
- estimated joint velocity
- estimated joint acceleration
- configured limit 초과
- TCP transform validity
- rotation matrix orthonormality
- determinant near +1
- gripper semantic 값
- gripper transition 최소 간격
- raw bag 존재

validation 결과는 JSON으로 저장하고 사람에게 요약한다.

잘못된 trajectory는 `--execute` replay를 거부한다.

---

## 13. UR pose와 SE(3) 변환

`core/transforms.py`에 구현한다.

### 13.1 convention

모든 transform은 다음 convention을 따른다.

```text
T_A_B = frame B의 좌표를 frame A로 변환
p_A = T_A_B @ p_B
```

예:

```text
T_base_tcp
T_tcp_camera
T_base_camera = T_base_tcp @ T_tcp_camera
```

### 13.2 필수 함수

```python
ur_pose_to_matrix(pose6) -> np.ndarray  # (4,4)
matrix_to_ur_pose(T) -> np.ndarray      # (6,)
quaternion_pose_to_matrix(position, quaternion_xyzw) -> np.ndarray
matrix_to_quaternion_pose(T) -> tuple[np.ndarray, np.ndarray]
validate_transform(T, atol=...) -> None
relative_pose_action(current_T, next_T, frame="tool") -> np.ndarray
apply_relative_pose_action(current_T, delta6, frame="tool") -> np.ndarray
```

### 13.3 UR rotation vector

```python
R.from_rotvec([rx, ry, rz]).as_matrix()
R.from_matrix(matrix).as_rotvec()
```

radian을 사용한다.

### 13.4 Tool/body-frame delta

```text
R_delta_tool = R_current^T @ R_next
p_delta_tool = R_current^T @ (p_next - p_current)
rotvec_delta_tool = log(R_delta_tool)
```

결과:

```text
[dx_tool, dy_tool, dz_tool, dRx_tool, dRy_tool, dRz_tool]
```

### 13.5 Base/spatial-frame delta

```text
p_delta_base = p_next - p_current
R_delta_base = R_next @ R_current^T
rotvec_delta_base = log(R_delta_base)
```

결과:

```text
[dx_base, dy_base, dz_base, dRx_base, dRy_base, dRz_base]
```

`T_next @ inverse(T_current)`의 translation column을 무조건 `p_next-p_current`로 취급하지 않는다.

### 13.6 round-trip 요구

다음 관계를 test한다.

```text
T_next_reconstructed = apply_relative_pose_action(T_current, delta)
T_next_reconstructed ~= T_next
```

Tool/base frame 각각 검증한다.

### 13.7 특이점과 표현 중복

rotation vector는 동일 회전에 여러 표현이 존재할 수 있다. rotvec 성분을 직접 비교하는 대신 rotation matrix 또는 relative rotation angle로 비교한다.

---

## 14. Replay

### 14.1 기본 안전 원칙

- `replay` 기본은 dry-run
- `--execute`에서만 actual trajectory와 gripper command 전송
- 시작 joint position이 tolerance 안에 없으면 실행 거부
- 자동으로 시작점까지 이동하지 않음
- validation fail이면 실행 거부
- safety normal이 아니면 실행 거부
- program running이 필요하도록 설정된 경우 false면 실행 거부
- Ctrl+C 시 goal cancel

### 14.2 dry-run

Dry-run에서 다음을 수행한다.

- trajectory validation
- planned duration
- point 수
- max velocity/acceleration
- gripper event timeline
- expected topics
- output episode path
- 실제 service/action 호출이 없음을 테스트

### 14.3 FollowJointTrajectory

configured action server를 사용한다.

```text
/scaled_joint_trajectory_controller/follow_joint_trajectory
```

- joint order 명시
- time_from_start 생성
- trajectory point monotonicity 보장
- timeout 처리
- goal accepted 확인
- feedback 수집
- result code 기록

### 14.4 Gripper replay timing

가능하면 FollowJointTrajectory feedback의 trajectory progress를 사용한다.

- feedback의 desired/actual `time_from_start` schema를 설치된 interface에서 확인
- controller progress가 다음 gripper event time을 통과할 때 command 실행
- feedback stale이면 abort
- 동일 event 중복 실행 방지
- planned time, trigger progress, command time, confirmation time, timing error 기록

wall-clock만 사용하는 구현은 speed scaling 또는 pause 시 drift가 생길 수 있으므로 기본 구현으로 사용하지 않는다. 설치된 controller feedback으로 신뢰성 있게 progress를 얻을 수 없는 경우 추측 구현을 하지 말고 `UNVERIFIED_ON_HARDWARE`로 보고하고 명시적 fallback을 별도 옵션으로 둔다.

### 14.5 초기 gripper command

trajectory 시작 전에 첫 semantic state를 DO1에 설정하고 readback 확인 후 configured settle time을 기다린다.

### 14.6 Replay rosbag

actual replay에서 trajectory 전송 전에 replay rosbag을 시작한다.

순서:

```text
preflight
-> replay rosbag start
-> topic readiness 확인
-> initial gripper command
-> trajectory goal send
-> gripper events
-> result wait
-> end settle
-> rosbag stop
-> summary/validation 저장
```

replay bag은 primary/wrist raw image와 actual robot state를 포함해야 한다.

### 14.7 Execution summary

최소 필드:

```text
execute
goal_accepted
result_code
planned_duration_sec
actual_duration_sec
joint_tracking_rmse
joint_tracking_max_error
gripper_event_count
gripper_event_timing_errors
camera_topic_message_counts
state_topic_message_counts
bag_storage_id
bag_exit_code
failure_reason
```

정교한 최종 품질 판정은 후속 브랜치에서 수행하되, collector가 계산 가능한 capture summary는 저장한다.

---

## 15. 시간과 동기화 데이터

향후 synchronization을 위해 가능한 모든 경계에서 다음 시간을 구분한다.

```text
ROS source/header time
local receipt ROS time
steady/monotonic time
episode elapsed time
```

wall-clock만 저장하지 않는다.

각 topic/message에 header stamp가 없으면 그 사실을 schema에 표시하고 receipt time을 사용한다.

NTP나 PTP 동기화가 되어 있다고 임의로 가정하지 않는다.

---

## 16. 테스트

ROS와 실제 로봇 없이 다음 테스트가 모두 실행되어야 한다.

```bash
pytest -q
python -m compileall collector
```

필수 테스트:

### Config

- 정상 config
- unknown key
- invalid gripper polarity
- invalid action mask length
- duplicate joint name
- invalid rate/timeout

### Transform

- identity pose
- pure translation
- pure rotation
- near-zero rotation
- near-pi rotation
- random pose matrix round-trip
- quaternion round-trip
- tool-frame delta round-trip
- base-frame delta round-trip
- invalid homogeneous matrix
- determinant/orthogonality validation

### Gripper core

- semantic 0/1 mapping
- physical polarity inversion
- invalid semantic value
- command-on-change-only
- minimum interval
- dry-run does not invoke adapter
- readback confirmation state machine

### Keyboard

- key mapping
- nonblocking command queue
- open/close/finish/abort event
- repeated key handling

### Trajectory

- joint reorder
- monotonic time
- duplicate timestamp
- NaN/Inf
- gripper transition extraction
- max velocity/acceleration
- initial joint tolerance

### Replay scheduler

- progress crossing event
- multiple events
- no duplicate command
- stale feedback abort
- cancel propagation

### Episode storage

- unique episode id
- overwrite refusal
- atomic status update
- failed/aborted raw data preservation

ROS adapter 테스트는 fake clients/messages 또는 dependency injection으로 분리한다. 일반 환경에서 `rclpy`가 없다는 이유로 pure test 전체가 import failure가 되면 안 된다.

---

## 17. 문서

### README.md

프로젝트 전체 파이프라인을 유지하면서 collector quick start를 추가한다.

### docs/collector_spec.md

최소한 다음을 포함한다.

- 실제 수집 workflow
- freedrive는 사용자가 활성화한다는 점
- 키보드 매핑
- `--execute` 의미
- DO1 semantic/polarity
- raw와 structured trajectory 차이
- episode directory 구조
- dry-run replay
- actual replay safety checklist
- troubleshooting

### docs/official_sources.md

공식 source와 프로젝트 고유 결정의 대응표를 작성한다.

각 결정은 다음 중 하나로 분류한다.

```text
OFFICIAL_OCTO
OFFICIAL_UR
PROJECT_DECISION
UNVERIFIED_ON_HARDWARE
```

### 하드웨어 검증 문서

다음 항목은 실제 UR5e에서 확인하기 전까지 `UNVERIFIED_ON_HARDWARE`다.

- 사용자의 freedrive 방식에서 SetIO service가 유지되는지
- DO1 polarity
- gripper 기계 동작 지연
- IOStates readback latency
- trajectory feedback progress와 gripper event timing
- primary/wrist topic 이름과 QoS
- MCAP plugin 설치 여부

---

## 18. Git 규칙

현재 branch는 반드시 다음이어야 한다.

```text
feature/collector
```

사용자 변경을 reset/revert하지 않는다.

권장 commit 단위:

```text
docs: add collector v2 implementation prompt
chore: configure collector package and cli
feat: add collector config and episode storage
feat: add ur pose transform utilities
feat: add digital output gripper control
feat: add freedrive demonstration recording
feat: add trajectory validation and replay
feat: add replay rosbag capture
test: add collector unit tests
docs: document collector workflow and official sources
```

실제 commit 메시지의 오타 없는 prefix를 사용한다. 위 예시의 ` test:` 앞 공백은 제거한다.

`.gitignore`에 최소한 다음을 반영한다.

```gitignore
data/raw/**
!data/raw/.gitkeep
data/rlds/**
!data/rlds/.gitkeep
runs/**
cache/**
wandb/**
*.db3
*.mcap
__pycache__/
.pytest_cache/
```

`metadata.yaml`을 전역 ignore하면 프로젝트 문서까지 예상치 못하게 무시할 수 있으므로 bag 경로 기준 pattern을 사용한다.

대용량 raw data, bag, checkpoint를 commit하지 않는다.

---

## 19. 완료 조건

다음 조건을 모두 만족해야 완료다.

### 패키징

```bash
python -m pip install -e .
octo-collector --help
```

성공해야 한다.

### CLI

다음 subcommand가 존재해야 한다.

```text
doctor
record-demo
validate-demo
replay
inspect
```

### 안전

- dry-run에서 action/service가 호출되지 않음
- `record-demo --execute`에서만 키보드 gripper command가 실제 DO1로 전달됨
- `replay --execute`에서만 trajectory와 gripper command가 실제 전달됨
- validation fail trajectory 실행 거부
- Ctrl+C goal cancel 및 raw data 보존

### 데이터

- demonstration raw bag 존재
- structured trajectory 존재
- gripper transition 존재
- replay raw bag 경로 지원
- manifest/status/config snapshot 존재
- raw를 덮어쓰지 않음

### 수학

- pose/matrix round-trip test 통과
- tool/base relative action round-trip test 통과
- invalid transform rejection test 통과

### 테스트

```bash
pytest -q
python -m compileall collector
```

통과해야 한다.

### 범위

RLDS, Octo training, inference, Docker GPU 학습 구현으로 작업을 확장하지 않는다.

---

## 20. 실제 하드웨어 명령에 대한 규칙

Codex는 실제 로봇을 움직이거나 DO 값을 변경하는 명령을 자동으로 실행하지 않는다.

다음 명령은 사용자에게 검증용으로 보여줄 수 있지만 직접 실행하지 않는다.

```bash
ros2 service call /io_and_status_controller/set_io \
  ur_msgs/srv/SetIO \
  "{fun: 1, pin: 1, state: 0.0}"

ros2 service call /io_and_status_controller/set_io \
  ur_msgs/srv/SetIO \
  "{fun: 1, pin: 1, state: 1.0}"
```

실제 replay도 사용자가 명시적으로 `--execute`를 사용해야만 가능하게 구현한다.

---

## 21. 작업 종료 보고 형식

작업이 끝나면 다음 순서로 보고한다.

1. 현재 branch
2. 구현한 기능 요약
3. 변경 파일 목록
4. CLI 사용 예시
5. test 결과
6. `git status --short`
7. commit 목록
8. 실제 하드웨어에서 아직 확인해야 할 항목
9. 후속 브랜치로 넘길 데이터 계약
10. 알려진 제한사항

최종 보고에는 다음 명령의 결과를 포함한다.

```bash
git branch --show-current
git status --short
git log --oneline --decorate -10
pytest -q
python -m compileall collector
octo-collector --help
```

하드웨어에서 검증하지 않은 기능을 검증 완료라고 표현하지 않는다.
