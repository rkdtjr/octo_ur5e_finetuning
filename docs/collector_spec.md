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

Before actual use confirm normal safety, program running, start joints, clear
workspace, E-stop access, camera topics/QoS, MCAP, DO0 polarity and gripper/IO
latency. Ctrl+C preserves raw files. For failures inspect `rosbag.log`,
`status.json`, and `validation.json`. TF/calibration warnings mean TCP accuracy
is not guaranteed.
