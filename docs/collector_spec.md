# UR5e Demonstration Collector v2

The operator enables freedrive externally and moves the arm by hand. The
collector does not manage robot power, brakes, PolyScope, freedrive, or
controller switching. Run `doctor` first.

During recording, `0`/`o` is open (semantic 0), `1`/`c` is closed (semantic 1),
`q` saves, and Escape aborts while preserving data. Without `--execute`, events
are recorded but SetIO is never called. With it, the configured Standard Digital
Output is used (DO0 on this hardware) with configurable polarity (verified
open=1, closed=0). IO readback confirms controller
output only—not finger width, contact, grasp success, or mechanical completion.

Raw topics are stored without image re-encoding in rosbag2. `samples.npz` is the
working record; `trajectory.npz` contains time, six joints, TCP rotation-vector
pose, and zero-order-held gripper semantics. Final synchronization and 10 Hz
dataset creation are deliberately deferred.

```text
manifest.json, status.json, config_resolved.yaml
demonstration/{rosbag2,events.jsonl,samples.npz,trajectory.npz,validation.json}
replay/{rosbag2,events.jsonl,execution_summary.json}
```

Replay is dry-run unless `--execute` is present. It refuses invalid trajectories
or a robot outside initial joint tolerance and never moves to the start
automatically. The current actual replay requires the explicit
`--wall-clock-gripper-fallback`; controller-feedback timing remains
`UNVERIFIED_ON_HARDWARE`, and wall-clock timing may drift under speed scaling.
Action completion follows controller feedback so UR speed scaling may extend
wall-clock duration. Once controller progress reaches planned duration, result
waiting is bounded by `result_timeout_margin_sec`. Only when feedback is absent
does the wall-clock fallback use
`planned_duration * result_timeout_factor + result_timeout_margin_sec`.
Timeout causes goal cancellation, bag shutdown, and a failed summary.

Before actual use confirm normal safety, program running, start joints, clear
workspace, E-stop access, camera topics/QoS, MCAP, DO0 polarity and gripper/IO
latency. Ctrl+C preserves raw files. For failures inspect `rosbag.log`,
`status.json`, and `validation.json`. TF/calibration warnings mean TCP accuracy
is not guaranteed.

The deployed WITHROBOT oCam-1CGN-U-T2 exposes 8-bit GRBG Bayer frames. The
`octo-wrist-camera` default uses its stable device link
`/dev/v4l/by-id/usb-WITHROBOT_Inc._oCam-1CGN-U-T2_SN_3AA01020-video-index0`.
It captures 1280x800 GRBG through `v4l2-ctl --stream-mmap` (30 FPS collector
default; the camera also supports the previously verified 60 FPS mode),
publishes the one-byte-per-pixel sensor stream on `/wrist_camera/image_raw` as
`bayer_grbg8`, and provides a 10 Hz debayered `bgr8` preview on
`/wrist_camera/image_color`. The replay recorder consumes only the Bayer source,
debayers it once, and sends every accepted 30 FPS frame to H.264; neither camera
topic nor the preview is duplicated in MCAP. Robot-state MCAP uses the
`zstd_fast` chunk-compression preset.

For replay capture, camera Image topics are not stored in MCAP. Primary
1280x720@30 and wrist 1280x800@30 frames are converted to BGR encoder input and
written as H.264 Matroska streams. FFmpeg probes `h264_nvenc` with a real encode
and falls back to `libx264`; the selected encoder is recorded in metadata.
Timestamp CSV rows are written only after the matching frame bytes have been
accepted by FFmpeg, so CSV row count and encoded frame count are 1:1. Queue-full
frames are omitted from CSV and counted in `quality_report.json`.

Robot state topics, including `/octo_collector/robot_state`, are stored in
`robot_states.mcap`. The JSON state topic carries ROS and monotonic timestamps,
actual joints/velocities, TCP position/quaternion, gripper semantic state,
replay phase, and desired controller joints when feedback provides them.
