# Codex 구현 지시: Octo UR5e Collector v2

당신은 ROS 2 Jazzy, Universal Robots ROS 2 Driver, 로봇 데이터 수집, 좌표변환, RLDS 및 Octo 파인튜닝 파이프라인에 익숙한 시니어 로보틱스 소프트웨어 엔지니어다.

현재 저장소 이름은 `octo_ur5e_finetuning`이다. 디렉터리 스캐폴드는 이미 생성되어 있으나 대부분의 Python 파일은 placeholder 상태다.

이번 작업의 직접 범위는 **Collector v1 구현**이다. 다만 이후 RLDS 변환과 Octo 파인튜닝에서 다시 뜯어고치지 않도록, 데이터 계약과 action 표현은 아래의 **Octo 공식 코드 및 예제와 호환되는 방향**으로 설계해야 한다.

---

## 0. 절대 원칙

1. 저장소 전체를 먼저 읽고 현재 구조, placeholder, 설정 파일, 테스트 파일을 확인한다.
2. 구현 전에 변경 예정 파일과 데이터 흐름을 15줄 이내로 요약한다.
3. 실제 UR5e를 움직이는 동작은 기본적으로 비활성화한다.
4. `--execute`가 없는 경우 절대로 trajectory goal이나 digital output command를 실제 로봇에 보내지 않는다.
5. 외부 인터페이스 이름을 코드에 하드코딩하지 않는다. ROS topic, service, action, frame, pin, camera 이름은 YAML 설정으로 둔다.
6. 수집한 raw 데이터는 임의로 버리거나 덮어쓰지 않는다.
7. 공식 예제와 프로젝트 고유 설계를 문서에서 구분한다.
8. 실제 하드웨어에서 확인하지 않은 항목은 `UNVERIFIED_ON_HARDWARE`로 명시한다.
9. 구현 과정에서 공식 코드를 그대로 대량 복사하지 말고, 인터페이스와 데이터 계약을 참고해 이 저장소 구조에 맞게 작성한다.
10. 공식 자료와 현재 설치된 패키지의 인터페이스가 다르면 추측하지 말고 차이를 보고한다.

---

# 1. 반드시 먼저 확인할 공식 자료

아래 자료를 작업 시작 전에 직접 열어 확인한다. 링크가 열리지 않거나 내용이 변경되어 요구사항과 충돌하면 추측으로 진행하지 말고 보고한다.

## 1.1 Octo 공식 저장소 및 파인튜닝 자료

### A. Octo 공식 저장소

- https://github.com/octo-models/octo

확인할 사항:

- 설치 방식
- pretrained checkpoint 로딩 방식
- 공식 `scripts/finetune.py` 실행 방식
- 공식 examples 목록
- Octo 1.5 checkpoint 명칭

### B. 공식 최소 파인튜닝 예제

- https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.py

이 예제는 다음을 보여주는 참고 자료다.

- 새 observation space 구성
- 새 action space 구성
- RLDS dataset을 `make_single_dataset`으로 읽는 방법
- pretrained config 수정
- 새 proprio tokenizer와 action head 구성
- pretrained parameter merge
- transformer freeze 선택
- train step 및 checkpoint 저장

중요:

- 이 예제는 simulated ALOHA용이다.
- 예제의 `action_horizon=50`, `action_dim=14`, `L1ActionHead`는 ALOHA 태스크 고유 설정이다.
- UR5e 프로젝트에 위 값을 그대로 복사하지 않는다.
- 이 프로젝트의 기본 action 후보는 7차원이다.

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

- action horizon은 데이터셋과 Octo 설정을 함께 검토하여 별도 설정값으로 둔다.

### C. 공식 고급 파인튜닝 스크립트

- https://github.com/octo-models/octo/blob/main/scripts/finetune.py

확인할 사항:

- `OctoModel.load_pretrained`
- `make_single_dataset`
- `process_text`
- model config update
- `merge_params`
- optimizer freeze key
- checkpoint와 finetune config 저장
- validation 및 visualization callback

### D. 공식 파인튜닝 설정

- https://github.com/octo-models/octo/blob/main/scripts/configs/finetune_config.py

특히 다음 항목을 확인한다.

- `image_obs_keys={"primary": ..., "wrist": ...}`
- `proprio_obs_key`
- `language_key`
- `action_normalization_mask`
- `standardize_fn`
- `window_size`
- `action_horizon`
- `full`, `head_only`, `head_mlp_only`
- `image_conditioned`, `language_conditioned`, `multimodal`
- primary/wrist resize 및 augmentation 설정

공식 설정에서 7차원 action의 마지막 gripper 차원을 normalization 대상에서 제외하는 예제가 있으므로, 이 프로젝트도 gripper normalization 여부를 명시적으로 설정하고 테스트한다.

```python
[True, True, True, True, True, True, False]
```

단, 위 mask는 action이 정확히 7차원일 때만 유효하다. action dimension을 바꾸면 자동으로 길이를 검증해야 한다.

### E. 공식 dataloader 예제

- https://github.com/octo-models/octo/blob/main/examples/05_dataloading.ipynb
- https://github.com/octo-models/octo/blob/main/octo/data/dataset.py

확인할 사항:

- RLDS trajectory가 Octo standardized format으로 변환되는 과정
- observation/action 키 요구사항
- image key mapping
- proprio key mapping
- language instruction mapping
- normalization statistics 생성 및 사용
- action pad mask와 timestep pad mask의 의미

### F. 공식 inference 예제

- https://github.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb
- https://github.com/octo-models/octo/blob/main/examples/03_eval_finetuned.py

Collector 단계에서 inference를 구현하지는 않지만, 최종 observation/action shape가 inference adapter와 일관되는지 확인하는 참고 자료로 사용한다.

## 1.2 Universal Robots ROS 2 공식 자료

### A. UR ROS 2 Driver controller 문서

- https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/jazzy/doc/ur_robot_driver/ur_controllers/doc/index.html

확인할 사항:

- `io_and_status_controller`
- `~/io_states [ur_msgs/msg/IOStates]`
- `~/set_io [ur_msgs/srv/SetIO]`
- robot program running 상태
- safety mode
- scaled trajectory controller 특성

### B. UR ROS 2 Driver 공식 예제

- https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/main/ur_robot_driver/examples/examples.py

`SetIO` 호출 방식과 `FollowJointTrajectory` 사용 방식의 참고 자료로만 사용한다.

## 1.3 공식 자료 고정 및 기록

작업 시작 시 다음을 수행한다.

1. 현재 Octo 공식 저장소 `main`의 commit SHA를 확인한다.
2. 현재 UR ROS 2 Driver `jazzy` branch의 commit SHA 또는 설치 패키지 버전을 확인한다.
3. `docs/official_sources.md`에 다음을 기록한다.

```text
source_name
url
checked_at_utc
commit_or_version
used_for
project_specific_difference
```

`main` branch 링크만 남기고 끝내지 말고, 구현 시점의 commit SHA도 기록해 재현성을 확보한다.

---

# 2. 이번 작업 범위

이번 단계에서 구현할 것은 다음이다.

1. Git 친화적인 Python 패키징과 CLI
2. config schema 및 validation
3. episode/manifest/status 관리
4. 프리드라이브 demonstration recorder
5. Digital Output 1 기반 gripper state 기록 및 replay command
6. UR TCP pose와 transformation matrix 변환 유틸리티
7. trajectory validation
8. dry-run replay 및 명시적 actual replay
9. replay 중 raw rosbag2 recording
10. raw data quality metadata
11. pure-Python unit tests
12. 문서화

이번 단계에서 구현하지 않는다.

- RLDS 최종 builder 완성
- Octo 실제 fine-tuning 실행
- inference node 완성
- Docker GPU 환경 최종 튜닝
- hand-eye calibration 수행
- 자동 freedrive 활성화
- 자동 robot power/brake release/program play

단, 이후 단계와 호환되도록 인터페이스와 placeholder 문서는 정리한다.

---

# 3. 목표 데이터 흐름

```text
[Freedrive demonstration]
        |
        | 100 Hz actual joint/TCP/DO state
        v
raw demonstration trajectory
        |
        | validate + build replay plan
        v
[Dry-run validation]
        |
        | explicit --execute only
        v
UR5e trajectory replay + Digital Output 1 transitions
        |
        | raw camera/state/TF/controller/IO recording
        v
rosbag2 + capture index + execution events
        |
        | future phase
        v
offline synchronization at camera timestamps
        |
        v
quality evaluation
        |
        v
RLDS conversion
        |
        v
Octo official dataloader compatibility check
        |
        v
Octo fine-tuning
```

중요:

- 수집 단계에서 억지로 10 Hz 샘플을 확정하지 않는다.
- 모든 raw state와 image message를 원본 timestamp와 함께 보존한다.
- 10 Hz dataset 생성은 오프라인 synchronization 단계에서 수행한다.

---

# 4. Git 작업 규칙

1. 현재 branch와 working tree를 확인한다.
2. 사용자의 기존 변경사항을 삭제하거나 reset하지 않는다.
3. 새 branch가 필요하면 다음 이름을 사용한다.

```text
feat/collector-v1
```

4. 커밋은 가능한 경우 아래 단위로 분리한다.

```text
chore: configure collector package and cli
feat: add episode storage and config validation
feat: add ur pose transformation utilities
feat: add digital output gripper adapter
feat: add demonstration recorder
feat: add trajectory validation and replay
feat: add replay rosbag recorder
 test: add collector core tests
docs: document collector workflow and official sources
```

5. large raw data, rosbag, checkpoints, cache, WandB output을 Git에 포함하지 않는다.
6. `.gitignore`에 최소한 다음을 확인한다.

```text
data/raw/**
data/rlds/**
runs/**
cache/**
wandb/**
*.db3
*.mcap
metadata.yaml
__pycache__/
.pytest_cache/
```

필요한 `.gitkeep`은 유지한다.

7. 작업 종료 시 다음을 보고한다.

- branch
- commit 목록 또는 아직 commit하지 않은 이유
- `git status --short`
- 변경 파일 요약

---

# 5. 패키징 및 CLI

현재 `collector/` 아래 패키지가 editable install과 console script로 실행되도록 구성한다.

필수 console script:

```toml
octo-collector = "octo_ur5e_collector.collector_cli:main"
```

ROS가 설치되지 않은 일반 Python 환경에서도 pure-Python 모듈과 테스트는 import 가능해야 한다.

따라서 다음을 분리한다.

```text
core/
  config.py
  episode.py
  schema.py
  trajectory.py
  transforms.py
  quality.py
  time_utils.py

ros_adapters/
  joint_state_source.py
  tf_source.py
  digital_output_gripper.py
  trajectory_action_client.py
  rosbag_recorder.py
```

실제 디렉터리 이름은 기존 구조에 맞게 조정할 수 있지만, pure core와 ROS adapter의 의존성 분리는 유지한다.

ROS package를 pip dependency로 추가하지 않는다.

---

# 6. Config 요구사항

`collector/config/collector.yaml`을 typed config로 읽고 startup 시 검증한다.

예시 구조:

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
  - logical_name: wrist
    image_topic: /wrist_camera/image_raw
    camera_info_topic: null

sampling:
  demonstration_rate_hz: 100.0
  target_dataset_rate_hz: 10.0

timeouts:
  state_stale_sec: 0.2
  tf_lookup_sec: 0.1
  service_wait_sec: 3.0
  action_wait_sec: 5.0

replay:
  initial_joint_tolerance_rad: 0.05
  speed_scale: 1.0
  max_joint_velocity_rad_s: null
  max_joint_acceleration_rad_s2: null
  execute_requires_program_running: true

storage:
  output_root: data/raw
  rosbag_storage_id: mcap
  overwrite: false

gripper:
  semantic:
    open: 0
    closed: 1
  backend: ur_standard_digital_output
  output_pin: 1
  output_value_for_open: 0.0
  output_value_for_closed: 1.0
  command_on_change_only: true
  minimum_command_interval_sec: 0.2
  readback_from_io_states: true

action:
  dimension: 7
  translation_unit: meter
  rotation_representation: rotation_vector
  delta_frame: tool
  gripper_index: 6
  normalization_mask: [true, true, true, true, true, true, false]
```

주의:

- dataset semantic은 항상 `0=open`, `1=closed`로 고정한다.
- 실제 Digital Output의 polarity는 `output_value_for_open`, `output_value_for_closed`로 분리한다.
- 실제 장비에서 HIGH가 open인지 close인지 코드에 암묵적으로 넣지 않는다.
- config에 두 output value가 동일하면 validation error다.
- output pin은 현재 프로젝트 기본값 1이지만 config로 둔다.
- action dimension과 normalization mask 길이가 다르면 startup에서 실패한다.

---

# 7. Digital Output 1 그리퍼 구현

기존 RG2 URScript 함수 호출 방식은 사용하지 않는다.

그리퍼 명령은 Universal Robots ROS 2 Driver의 다음 service를 사용한다.

```text
/io_and_status_controller/set_io
ur_msgs/srv/SetIO
```

요청 의미:

```text
fun = 1      # set digital output
pin = 1      # configurable, default 1
state = 0.0 or 1.0
```

## 7.1 State 기록

`/io_and_status_controller/io_states`의 `ur_msgs/msg/IOStates`에서 config의 Standard Digital Output pin 값을 읽는다.

저장 시 다음을 구분한다.

```text
gripper_semantic_state       # 0=open, 1=closed
digital_output_pin
digital_output_value         # physical output 0/1
digital_output_source_time_ns
digital_output_receipt_time_ns
digital_output_age_ms
```

기계식 센서가 없으므로 다음 이름을 사용하지 않는다.

```text
actual_gripper_width
object_detected
mechanical_gripper_state
```

Digital Output readback은 controller output state일 뿐, 손가락이 실제로 완전히 움직였거나 물체를 잡았음을 보장하지 않는다. 이 한계를 manifest와 문서에 기록한다.

## 7.2 Replay command

- semantic state가 변할 때만 `SetIO`를 호출한다.
- 동일 state 반복 호출은 하지 않는다.
- minimum command interval을 검사한다.
- service unavailable, timeout, `success=false`를 실패로 처리한다.
- 실패 시 trajectory execution을 cancel하도록 상위 replay coordinator에 알린다.
- dry-run에서는 service client를 생성할 수 있지만 request를 호출하면 안 된다.
- command event마다 planned time, call time, response time, success를 기록한다.

예시 event:

```json
{
  "event": "gripper_command",
  "planned_elapsed_sec": 2.4,
  "semantic_state": 1,
  "pin": 1,
  "output_value": 1.0,
  "service_called": true,
  "success": true
}
```

## 7.3 Replay timing

trajectory point를 하나의 큰 FollowJointTrajectory goal로 전송하면서 gripper event를 별도 timer로 실행할 경우, speed scaling과 pause 때문에 planned wall-clock time과 실제 robot trajectory progress가 달라질 수 있다.

따라서 v1에서는 다음을 명시한다.

- demonstration elapsed time 기반 event scheduling을 구현한다.
- `speed_scale`은 동일하게 반영한다.
- robot pause 또는 speed scaling이 발생하면 동기화 오차가 생길 수 있음을 quality metric으로 기록한다.
- controller state에서 desired/actual progress를 신뢰성 있게 계산할 수 있다면 보정 기능을 별도 옵션으로 구현할 수 있다.
- 검증되지 않은 progress tracking을 임의로 구현하지 않는다.

---

# 8. UR pose와 행렬변환

좌표변환은 pure-Python core로 구현하고 철저히 테스트한다.

## 8.1 UR pose 표현

UR TCP pose가 다음 형태일 경우:

```text
[x, y, z, rx, ry, rz]
```

- `[x, y, z]`: meter
- `[rx, ry, rz]`: Euler angle이 아니라 rotation vector / axis-angle vector

다음 변환 함수를 제공한다.

```python
ur_pose_to_matrix(pose6) -> 4x4 homogeneous matrix
matrix_to_ur_pose(T) -> pose6
quaternion_pose_to_matrix(position, quaternion_xyzw) -> T
matrix_to_quaternion_pose(T) -> position, quaternion_xyzw
validate_transform(T)
```

rotation vector 변환은 직접 근사식을 새로 쓰지 말고 검증된 라이브러리를 사용한다.

권장:

```python
scipy.spatial.transform.Rotation
```

## 8.2 Frame convention

모든 transform 이름은 다음 규칙을 사용한다.

```text
T_A_B = frame B의 pose를 frame A에서 표현한 transform
p_A = T_A_B @ p_B
```

예:

```text
T_base_tcp
T_tcp_camera
T_base_camera = T_base_tcp @ T_tcp_camera
```

이 convention을 docstring과 테스트에서 고정한다.

## 8.3 Relative action

두 시점의 pose가 있을 때:

```text
T_base_tcp_t
T_base_tcp_next
```

### Tool/body frame delta

```text
T_delta_tool = inverse(T_base_tcp_t) @ T_base_tcp_next
```

이를 6D로 변환할 때:

```text
dp_tool = R_t^T @ (p_next - p_t)
dR_tool = R_t^T @ R_next
drotvec_tool = log(dR_tool)
```

### Base/spatial frame delta

base frame translation delta는 다음과 같이 계산한다.

```text
dp_base = p_next - p_t
```

rotation delta는 다음과 같이 계산한다.

```text
dR_base = R_next @ R_t^T
drotvec_base = log(dR_base)
```

주의:

- base-frame translation delta를 단순히 `T_next @ inverse(T_current)`의 translation column으로 사용하지 않는다.
- 그 translation column은 일반적인 `p_next - p_current`와 동일하지 않을 수 있다.

함수 예시:

```python
relative_pose_action(current_T, next_T, frame="tool") -> np.ndarray shape (6,)
apply_relative_pose_action(current_T, delta6, frame="tool") -> next_T
```

기본 `delta_frame`은 현재 프로젝트 결정으로 `tool`을 사용하되, config에서 `base`도 지원한다. manifest와 RLDS metadata에 반드시 frame convention을 기록한다.

## 8.4 Action 구성

기본 action:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

- translation: meter
- rotation: rotation vector, radian magnitude
- gripper: semantic binary 0/1
- gripper는 normalization에서 제외

action은 raw recording 중 생성하지 않는다. raw에는 absolute state를 저장하고, `processing/build_actions.py`에서 synchronized samples로부터 생성한다.

raw 보존 항목:

```text
joint position
joint velocity if available
joint effort if available
TCP translation
TCP quaternion
optional UR rotvec representation
Digital Output pin/value
semantic gripper state
all timestamps
```

## 8.5 Transform tests

최소 테스트:

- identity round trip
- random pose matrix round trip
- pure translation
- pure rotation
- near-zero rotation vector
- rotation close to pi
- invalid matrix shape
- non-orthonormal rotation rejection
- determinant not close to +1 rejection
- tool-frame delta reconstruction
- base-frame delta reconstruction
- camera chain multiplication order
- quaternion xyzw convention

허용오차를 명시하고 random seed를 고정한다.

---

# 9. Episode 데이터 계약

원시 episode 구조:

```text
data/raw/<episode_id>/
├── manifest.yaml
├── status.json
├── config_snapshot.yaml
├── logs/
│   ├── collector.log
│   ├── replay.log
│   └── rosbag.log
├── demonstration/
│   ├── trajectory.jsonl
│   ├── gripper_events.jsonl
│   └── rosbag2/
└── replay/
    ├── replay_plan.jsonl
    ├── execution_events.jsonl
    ├── capture_index.jsonl
    └── rosbag2/
```

## 9.1 manifest 필수 정보

- schema version
- episode id
- language instruction
- UTC creation time
- host name
- git commit hash
- git dirty 여부
- ROS distro
- Python version
- package versions where available
- robot name
- frame convention
- joint order
- camera logical names/topics
- gripper semantic mapping
- Digital Output pin and polarity mapping
- action representation candidate
- demonstration status
- replay status
- source files and sizes
- completed files SHA-256
- limitations

## 9.2 status state machine

```text
created
recording_demo
demo_recorded
validating_replay
replay_ready
replaying
replay_recorded
failed
aborted
```

- atomic write
- `.in_progress` marker
- invalid state transition rejection
- error reason 저장
- interrupted state recovery

## 9.3 trajectory.jsonl sample

최소 필드:

```text
sequence
source_timestamp_ns
receipt_timestamp_ns
monotonic_timestamp_ns
elapsed_sec
joint_names
joint_position_rad
joint_velocity_rad_s or null
joint_effort or null
tcp_translation_m
tcp_quaternion_xyzw
tcp_rotation_vector_rad optional
tcp_source_timestamp_ns
tcp_age_ms
gripper_semantic_state
digital_output_pin
digital_output_value
digital_output_source_timestamp_ns
digital_output_age_ms
validity_flags
```

- NaN/Infinity 금지
- unavailable은 null
- source time을 얻을 수 없으면 null + flag

---

# 10. Demonstration recorder

`record_demonstration_node.py`를 구현한다.

- 100 Hz steady/monotonic timer 기준으로 latest state를 sample한다.
- `/joint_states`를 config joint order로 재정렬한다.
- missing/duplicate joint를 검출한다.
- TCP는 TF base→tcp를 사용한다.
- gripper는 IOStates의 configured pin readback을 사용한다.
- source timestamp, receipt ROS time, monotonic time을 구분한다.
- freedrive를 자동으로 on/off하지 않는다.
- 빈 trajectory와 지나치게 짧은 trajectory를 정상 완료 처리하지 않는다.
- Ctrl+C/SIGTERM에서 flush하고 status를 명확히 남긴다.
- 기존 episode를 기본적으로 덮어쓰지 않는다.

프리드라이브 중 Digital Output 1이 바뀌면 gripper transition event도 별도 기록한다.

---

# 11. Replay validation

pure-Python validation 모듈을 구현한다.

검사항목:

- joint names/order
- sample 수
- timestamp monotonicity
- finite values
- joint limits if configured
- finite-difference velocity
- finite-difference acceleration
- initial actual joint tolerance
- state freshness
- gripper semantic 0/1
- Digital Output value mapping
- gripper transition minimum interval
- action server existence
- SetIO service existence
- robot program running if required
- safety mode if available

validation result를 machine-readable JSON과 human-readable summary로 모두 제공한다.

---

# 12. Replay 실행

`replay_trajectory_node.py`를 구현한다.

- 기본은 dry-run
- 실제 실행은 `--execute`
- `--dry-run`과 `--execute` 동시 사용 금지
- original elapsed time 유지
- replay speed scale 적용
- 임의 spline overshoot 생성 금지
- action rejection/abort/cancel/timeout 기록
- Ctrl+C 시 cancel 요청
- start joint tolerance 실패 시 실행 거부
- gripper transitions는 Digital Output service로 실행
- gripper command 실패 시 trajectory cancel
- 실제 실행 전 최종 validation summary를 출력

프리드라이브 trajectory의 모든 100 Hz 점을 그대로 action goal에 넣는 것이 controller와 네트워크에 적절한지 검토한다. 필요하면 tolerance 내에서 trajectory simplification/downsampling을 별도 pure function으로 제공하되:

- 기본은 원본 보존
- simplification은 opt-in
- 최대 pose/joint reconstruction error를 검증
- raw trajectory를 절대 수정하지 않음

---

# 13. Replay-time raw recorder

`record_replay_dataset_node.py`를 구현한다.

rosbag2에 최소 다음을 저장한다.

- joint states
- TF
- TF static
- trajectory controller state
- IO states
- robot program running
- safety mode
- primary image and camera info
- wrist image and camera info when configured

요구사항:

- topic 목록 config 기반
- 원본 message type/timestamp 유지
- image decode 및 PNG/JPEG 재저장 금지
- recorder ready 전 replay 시작 금지
- recorder failure 시 replay cancel
- 정상 종료 후 metadata 검증
- stdout/stderr log 보존
- capture index에는 lightweight metadata만 기록

---

# 14. CLI

최소 명령:

```bash
octo-collector doctor --config collector/config/collector.yaml

octo-collector record-demo \
  --config collector/config/collector.yaml \
  --episode-id demo_0001 \
  --instruction "pick up the blue object"

octo-collector validate-replay \
  --config collector/config/collector.yaml \
  --episode-id demo_0001

octo-collector replay \
  --config collector/config/collector.yaml \
  --episode-id demo_0001 \
  --dry-run

octo-collector record-replay \
  --config collector/config/collector.yaml \
  --episode-id demo_0001 \
  --execute

octo-collector inspect \
  --episode-id demo_0001
```

`doctor` 검사:

- config schema
- output directory
- ROS topic existence
- message types
- joint names
- TF lookup
- action server
- IO states topic
- SetIO service
- Digital Output pin existence
- camera topics
- robot program running state
- safety state

---

# 15. 향후 RLDS/Octo 호환성 계약

이번 단계에서 RLDS를 완성하지 않지만, `docs/rlds_contract.md`를 추가해 다음을 명시한다.

예상 standardized trajectory:

```python
{
    "observation": {
        "image_primary": ...,
        "image_wrist": ...,
        "proprio": ...,
        "timestep_pad_mask": ...,
    },
    "action": ...,
    "language_instruction": ...,
}
```

실제 RLDS builder의 원본 key와 Octo standardized key는 분리해서 문서화한다.

UR5e 후보 mapping:

```text
image_obs_keys:
  primary: image_primary
  wrist: image_wrist

proprio_obs_key: proprio
language_key: language_instruction
action_dim: 7
action_normalization_mask:
  [true, true, true, true, true, true, false]
```

중요:

- 공식 `bridge_dataset_transform`를 UR5e 데이터에 그대로 사용하지 않는다.
- UR5e용 `standardize_fn`은 이후 별도 구현한다.
- Collector raw schema와 RLDS schema를 동일시하지 않는다.
- RLDS action horizon은 Collector에서 고정하지 않는다.
- Octo 공식 advanced config의 기본 horizon 4와 ALOHA minimal example horizon 50을 구분한다.

`training/check_dataset.py`의 향후 목표도 문서에 남긴다.

- one batch load
- observation keys
- image dtype/shape
- proprio shape
- action shape
- finite values
- gripper unique values
- normalization statistics
- action pad mask
- timestep pad mask
- text task presence

---

# 16. 테스트

ROS/robot/camera 없이 실행되는 pure-Python 테스트를 우선한다.

최소 테스트:

## Config/schema

- required key
- invalid output polarity
- invalid pin
- normalization mask length
- unsupported delta frame

## Episode

- atomic status write
- invalid transition
- interrupted recovery
- no overwrite
- checksum

## Joint trajectory

- reorder
- missing/duplicate joint
- timestamp monotonicity
- velocity/acceleration limits
- initial tolerance

## Gripper

- physical output↔semantic mapping
- transition-only command
- repeated state ignored
- minimum interval
- dry-run no call
- service failure propagation

## Transform

- section 8.5 전체

## Serialization

- NaN/Infinity rejection
- null handling
- deterministic JSONL field behavior

## ROS adapter

- mock/fake only
- no real action/service call

---

# 17. 문서

다음을 실제 사용 가능한 수준으로 갱신한다.

```text
README.md
docs/collector_spec.md
docs/official_sources.md
docs/rlds_contract.md
docs/safety_checklist.md
```

README에 포함:

- 프로젝트 단계
- 설치
- ROS source
- config
- demo recording
- dry-run
- actual replay
- raw storage
- known limitations
- next phase

Safety checklist에 포함:

- workspace clear
- emergency stop access
- speed slider low
- correct robot
- correct Digital Output pin
- polarity manually verified
- start joint tolerance
- external control program running
- camera recording ready
- dry-run completed

---

# 18. 완료 기준

다음을 모두 수행하고 결과를 보고한다.

1. `python -m compileall` 성공
2. `pytest -q` 성공
3. editable install 성공
4. `octo-collector --help` 성공
5. ROS가 없는 환경에서 pure-Python 테스트 성공
6. dry-run이 action/service request를 실제 전송하지 않는 테스트 성공
7. transform round-trip 및 delta reconstruction 테스트 성공
8. Digital Output semantic mapping 테스트 성공
9. 공식 source URL 및 commit/version 기록
10. 변경 파일 목록
11. 실행 명령
12. 현재 제한사항
13. `UNVERIFIED_ON_HARDWARE` 목록
14. `git status --short`

실제 UR5e가 없으면 실제 동작 성공을 주장하지 않는다.

---

# 19. 작업 종료 보고 형식

아래 형식으로 답한다.

```text
## Summary

## Official references checked
- Octo commit:
- UR driver version/commit:

## Files changed

## Data and action conventions
- gripper semantic:
- physical Digital Output mapping:
- transform convention:
- delta frame:
- action dimension:

## Tests
- compileall:
- pytest:
- CLI:

## Commands to run

## Safety behavior

## UNVERIFIED_ON_HARDWARE

## Git status
```
