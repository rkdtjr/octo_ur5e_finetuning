# UR5e Episode Processing

This folder is self-contained and does not require ROS or the collector Python
package. It performs only offline processing:

```text
replay camera timestamp CSV + robot_states.mcap
        -> 10 Hz timestamp alignment
        -> joint/TCP/gripper interpolation
        -> 7D Cartesian actions
        -> synchronized_episode.npz + quality report
```

It does **not** create RLDS/TFRecord files. The RLDS builder should only decode
the selected video frames and package these already-computed arrays.

## Dependencies

```bash
pip install "numpy<2" scipy PyYAML
```

OpenCV is not required for synchronization/action processing. The optional
`processing.convert_to_rlds.decode_selected_rgb` boundary helper imports OpenCV
only when a later RLDS builder asks it to decode selected frames.

Quality reports retain the detailed `ACCEPTED/WARN_ACCEPTED/REJECT` result and
also expose the collector-compatible `GOOD/WARNING/BAD` `quality_grade`.

## Process one episode

After installing the project in the virtual environment, run from any directory:

```bash
octo-process-episode \
  data/raw/20260731_184359_c25768c6
```

Outputs are written to `<episode>/processed/`:

```text
processed/
├── synchronized_episode.npz
├── synchronization_index.csv
├── processing_manifest.yaml
├── action_statistics.json
└── quality_report.json
```

The output also contains `segment_id`, `segment_is_first`, and
`segment_is_last`. Invalid transitions are assigned `segment_id=-1`; the RLDS
builder should package each remaining contiguous segment independently rather
than deleting isolated steps and breaking trajectory continuity.

The first diagnostic training can use only the primary camera while retaining
wrist indices for later use. To require valid wrist frames for every step:

```bash
python -m processing.synchronize_episode EPISODE --require-wrist
```

## Process all episodes in a directory

```bash
octo-process-batch data/raw \
  --summary data/processing_summary.json
```

## Action contract

Each step stores:

```text
[dx, dy, dz, drx, dry, drz, gripper]
```

- Cartesian delta is `observation[t] -> observation[t+1]`.
- Default frame is the current TCP/tool frame.
- Translation is meters.
- Rotation is a rotation vector in radians.
- Gripper is absolute semantic next-state: `0=open`, `1=closed`.
- The gripper dimension should not be normalized by Octo.

The official Octo loader chunks present and future per-step actions later;
processing therefore stores one 7D action per transition, not a prebuilt action
horizon.

## Important timestamp rule

Frame selection uses `primary_timestamps.csv` and `wrist_timestamps.csv` ROS
header timestamps. A nominal 10 Hz grid selects unique primary frames, then the
selected primary frame timestamps become the actual observation timeline used
for robot interpolation and action generation. MKV presentation timestamps are
not used. The interval is cropped to `replay_state == "executing"`, excluding
preflight, settling, and return-to-start motion.
