"""Validate an Octo-compatible local TFDS/RLDS dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


def inspect_dataset(dataset_dir: str | Path) -> dict:
    builder = tfds.builder_from_directory(str(Path(dataset_dir)))
    report = {"dataset_dir": str(Path(dataset_dir).resolve()), "splits": {}}
    episode_ids: dict[str, set[str]] = {}
    all_actions = []
    for split in ("train", "val"):
        if split not in builder.info.splits:
            raise ValueError(f"missing required split: {split}")
        ids, lengths = set(), []
        for episode in builder.as_dataset(split=split):
            episode_id = episode["episode_metadata"]["episode_id"].numpy().decode()
            steps = list(episode["steps"].as_numpy_iterator())
            if len(steps) < 2:
                raise ValueError(f"{split}/{episode_id} has fewer than 2 steps")
            if not steps[0]["is_first"] or not steps[-1]["is_last"]:
                raise ValueError(f"{split}/{episode_id} has invalid RLDS boundary flags")
            actions = np.stack([step["action"] for step in steps])
            if actions.shape[1:] != (7,) or not np.isfinite(actions).all():
                raise ValueError(f"{split}/{episode_id} has an invalid action tensor")
            ids.add(episode_id); lengths.append(len(steps)); all_actions.append(actions)
        episode_ids[split] = ids
        report["splits"][split] = {
            "trajectories": len(lengths), "episodes": len(ids),
            "steps": int(sum(lengths)), "min_steps": min(lengths), "max_steps": max(lengths),
            "episode_ids": sorted(ids),
        }
    overlap = sorted(episode_ids["train"] & episode_ids["val"])
    if overlap:
        raise ValueError(f"episode leakage across train/val: {overlap}")
    actions = np.concatenate(all_actions)
    report["action"] = {
        "dimension": 7,
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
    }
    report["status"] = "GOOD"
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="octo-check-dataset")
    parser.add_argument("dataset_dir")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(inspect_dataset(args.dataset_dir), indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
