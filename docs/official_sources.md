# Official sources and project decisions

Checked at UTC: `2026-07-30`. GitHub pages did not expose a reliable immutable
SHA in the unauthenticated rendered view, so the reviewed branch is recorded
where a commit could not be established. Re-check before changing the contract.

| classification | source_name | url | branch_or_commit | installed_version | used_for | project_specific_difference |
|---|---|---|---|---|---|---|
| OFFICIAL_OCTO | Octo repository | https://github.com/octo-models/octo | main (SHA unavailable) | n/a | `OctoModel.load_pretrained`, `hf://rail-berkeley/octo-small-1.5`, finetune entrypoint/modes | Collector emits no final Octo action |
| OFFICIAL_OCTO | New observation/action example | https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.py | main (SHA unavailable) | n/a | `make_single_dataset`, observation/proprio/language keys, new head | ALOHA horizon 50/action 14/L1 head are not copied |
| OFFICIAL_OCTO | Fine-tune config | https://github.com/octo-models/octo/blob/main/scripts/configs/finetune_config.py | main (SHA unavailable) | n/a | window/horizon, standardization, normalization mask and modes | UR action contract is 7D and mask length is validated |
| OFFICIAL_OCTO | Dataset implementation | https://github.com/octo-models/octo/blob/main/octo/data/dataset.py | main (SHA unavailable) | n/a | RLDS loading/data transformations | RLDS work is deferred |
| OFFICIAL_UR | UR ROS 2 Driver controllers | https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/jazzy/doc/ur_robot_driver/ur_controllers/doc/index.html | Jazzy docs | ur_robot_driver 3.8.0 | GPIO topics/SetIO and trajectory controller | DO1 polarity is configuration |
| OFFICIAL_UR | Driver source | https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/tree/jazzy | jazzy (SHA unavailable) | 3.8.0 | adapter naming and controller behavior | freedrive/controller switching is external |
| OFFICIAL_UR | UR axis-angle representation | https://www.universal-robots.com/articles/ur/programming/axis-angle-representation/ | current article | n/a | TCP `[x,y,z,rx,ry,rz]` rotation vector | transform convention is explicitly `T_A_B` |
| OFFICIAL_UR | Installed SetIO/IOStates interfaces | local `ros2 interface show` | Jazzy packages | ur_msgs 2.5.0 | `FUN_SET_DIGITAL_OUT=1`, pin/state fields, digital output readback | readback is not mechanical state |
| OFFICIAL_UR | Installed FollowJointTrajectory | local `ros2 interface show` | Jazzy packages | control_msgs 5.9.0 | desired/actual feedback has `time_from_start` | feedback timing remains hardware-unverified |
| PROJECT_DECISION | Collector v2 contract | `collector/config/collector.yaml` | schema 2 | collector 0.2.0 | 7D future action, raw-first capture, episode status | final 10 Hz synchronization is deferred |
| OFFICIAL_UR | rosbag MCAP plugin | local `ros2 pkg prefix rosbag2_storage_mcap` | Jazzy | installed under `/opt/ros/jazzy` | raw bag storage | runtime write/message capture still needs hardware test |
| PROJECT_DECISION | SciPy Rotation | https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html | current docs | scipy 1.16.3, numpy 2.2.6 | rotvec/matrix/quaternion conversions | dependency supports older compatible versions |

## UNVERIFIED_ON_HARDWARE

- SetIO availability during the chosen external freedrive method
- physical DO1 polarity, gripper delay, and IOStates readback latency
- program/safety topic values and controller feedback progress during scaling
- primary/wrist topic names, message counts, and QoS
- MCAP recording with the deployed camera bandwidth
- TCP accuracy under the actual calibration
