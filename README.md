# Octo UR5e Fine-tuning

Pipeline:
1. Freedrive demonstration recording
2. Joint trajectory replay
3. Replay-time synchronized image/state recording
4. Episode quality evaluation
5. Action generation
6. RLDS conversion
7. Docker-based Octo fine-tuning
8. Offline validation
9. UR5e deployment

## Collector v2 quick start

By default the operator enables freedrive externally. With
`record-demo --enable-freedrive`, the collector switches from the configured
trajectory controller to the UR freedrive controller, publishes its keepalive,
and restores the trajectory controller when recording ends. The collector never
manages power, brakes, or PolyScope programs.

The robot must be powered, brakes released, safety mode normal, and the
PolyScope External Control program running before managed freedrive is started.
Check this without switching controllers:

```bash
octo-collector doctor --config collector/config/collector.yaml --freedrive
```

```bash
python -m pip install -e .
octo-collector doctor --config collector/config/collector.yaml
octo-collector record-demo --config collector/config/collector.yaml --instruction "pick up the blue object" --enable-freedrive
octo-collector validate-demo data/raw/<episode_id>
octo-collector replay data/raw/<episode_id>
```

All commands are safe/dry-run by default. `record-demo --execute` enables only
keyboard commands to the configured digital output (DO0 on this hardware);
only `replay --execute` may command arm motion. Read
`docs/collector_spec.md` before hardware use.

## Replay capture format

Actual replay stores both 30 FPS cameras as H.264 Matroska video rather than ROS
Image messages in MCAP. Matroska (`.mkv`) is the default because an interrupted
recording is generally more recoverable than MP4, whose final index is normally
written on clean close. Each accepted encoded frame has exactly one row in its
timestamp CSV; a frame rejected by a full encoder queue is counted as dropped
and receives no CSV row. Preview images are never recorded.

```text
replay/
  metadata.json
  robot_states.mcap
  robot_states_metadata.yaml
  primary.mkv
  wrist.mkv
  primary_timestamps.csv
  wrist_timestamps.csv
  quality_report.json
  episode_result.json
```

At conversion time, `processing/synchronize_episode.py` creates a common 10 Hz
timeline and selects the nearest unique source frames by ROS timestamp. It does
not use frame-index modulo downsampling. Video is decoded to RGB uint8 only for
selected samples; resize/crop remains a downstream configuration decision.

`quality_report.json` includes an explainable `evaluation` block whose
`overall` value is `GOOD`, `WARNING`, or `BAD`. Each check records its measured
value and thresholds. `execution_summary.json` repeats the compact
`quality_grade`, `quality_verdict`, and `quality_problems` fields.

## Processing and RLDS

Use `octo-dataset doctor data/raw` to inspect readiness. `octo-dataset build`
is dry-run by default; with `--execute` it incrementally processes missing
episodes, keeps `GOOD/WARNING` valid segments, validates one decoded trajectory,
and writes the TFDS/RLDS dataset. See `rlds_builder/README.md` for the schema and
training-workstation dependencies.

### Fine-tuning

Build an episode-level split so demonstrations from one recording never occur
in both train and validation:

```bash
octo-dataset build data/raw --output data/rlds/ur5e_pick_3val \
  --include-grades GOOD,WARNING --val-episodes 3 --execute
octo-check-dataset data/rlds/ur5e_pick_3val/ur5e_pick/1.0.0
```

The training integration uses the official Octo fine-tuning entrypoint,
`octo-small-1.5`, a two-frame history, four-action horizon, and 7D UR5e action
contract. It defaults to `head_only` and the primary camera for the small-data
diagnostic. W&B is disabled for smoke runs and offline for longer local runs.

```bash
octo-train --smoke
octo-train --execute --smoke
octo-train --execute
octo-plot-metrics runs/octo_ur5e/pick/<run_name>
```
