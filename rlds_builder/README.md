# UR5e RLDS Pipeline

The dataset pipeline is incremental and safe by default:

```text
raw collector episode
  -> processed/synchronized_episode.npz
  -> quality and valid-segment selection
  -> selected RGB frame decode and 256x256 preprocessing
  -> TFDS/RLDS train + val dataset
```

## Robot workstation

The robot workstation can inspect, process, and plan without TensorFlow:

```bash
octo-dataset doctor data/raw
octo-dataset process data/raw
octo-dataset process data/raw --execute
octo-dataset build data/raw --output data/rlds/ur5e_pick
```

`build` is a dry-run unless `--execute` is passed. It reports pending/reused
episodes and missing build dependencies.

## Training workstation

```bash
pip install -e '.[rlds]'
octo-dataset build data/raw \
  --output data/rlds/ur5e_pick \
  --val-episodes 3 \
  --include-grades GOOD,WARNING \
  --execute
```

Each contiguous valid segment becomes one independent RLDS trajectory. Invalid
transitions, preflight footage, final settling, and return-to-start motion are
never packaged as training steps. Existing non-empty output directories are not
overwritten.

The raw step schema contains `observation.image_primary`,
`observation.image_wrist`, 14D `observation.proprio`, 7D `action`,
`language_instruction`, and the three RLDS boundary flags. For Octo, use image
keys `primary` and `wrist`, proprio key `proprio`, language key
`language_instruction`, and action normalization mask
`[true, true, true, true, true, true, false]`.
