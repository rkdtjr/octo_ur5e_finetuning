"""Generate 7D Octo action transitions from synchronized reached TCP poses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .transforms import quaternion_pose_to_matrix, relative_pose_action
except ImportError:  # direct script execution
    from transforms import quaternion_pose_to_matrix, relative_pose_action

ACTION_DIMENSION = 7
ACTION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")


def build_relative_actions(
    tcp_positions: np.ndarray,
    tcp_quaternions_xyzw: np.ndarray,
    gripper_states: np.ndarray,
    *,
    frame: str = "tool",
    gripper_target: str = "next",
) -> np.ndarray:
    """Build one action for each ``observation[t] -> observation[t+1]``.

    The motion part is the reached-pose SE(3) delta. The gripper component is
    absolute semantic state, not a delta. ``gripper_target='next'`` aligns a
    close command with the transition that changes open -> closed, which is the
    appropriate convention for learning from state-only replay logs.
    """

    positions = np.asarray(tcp_positions, dtype=np.float64)
    quaternions = np.asarray(tcp_quaternions_xyzw, dtype=np.float64)
    gripper = np.asarray(gripper_states, dtype=np.int8)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("tcp_positions must have shape [N, 3]")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("tcp_quaternions_xyzw must have shape [N, 4]")
    if gripper.shape != (len(positions),):
        raise ValueError("gripper_states must have shape [N]")
    if len(positions) < 2:
        raise ValueError("at least two synchronized observations are required")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("TCP poses must be finite")
    if not np.isin(gripper, [0, 1]).all():
        raise ValueError("gripper_states must contain semantic 0/1")
    if gripper_target not in {"current", "next"}:
        raise ValueError("gripper_target must be 'current' or 'next'")

    transforms = [
        quaternion_pose_to_matrix(position, quaternion)
        for position, quaternion in zip(positions, quaternions)
    ]
    actions = np.empty((len(positions) - 1, ACTION_DIMENSION), dtype=np.float32)
    for index in range(len(actions)):
        actions[index, :6] = relative_pose_action(
            transforms[index], transforms[index + 1], frame=frame
        )
    actions[:, 6] = (
        gripper[1:] if gripper_target == "next" else gripper[:-1]
    ).astype(np.float32)
    return actions


def build_relative_actions_masked(
    tcp_positions: np.ndarray,
    tcp_quaternions_xyzw: np.ndarray,
    gripper_states: np.ndarray,
    observation_valid: np.ndarray,
    *,
    frame: str = "tool",
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite actions only where both endpoint observations are valid.

    Invalid transitions receive an all-zero placeholder and must be excluded by
    the returned mask. This preserves timeline continuity and lets downstream
    code split valid contiguous segments without deleting timestamps.
    """
    positions=np.asarray(tcp_positions);quaternions=np.asarray(tcp_quaternions_xyzw)
    gripper=np.asarray(gripper_states);valid=np.asarray(observation_valid,dtype=bool)
    if valid.shape!=(len(positions),):raise ValueError("observation_valid shape mismatch")
    transition_valid=valid[:-1]&valid[1:]
    actions=np.zeros((max(0,len(positions)-1),ACTION_DIMENSION),dtype=np.float32)
    for index in np.flatnonzero(transition_valid):
        actions[index]=build_relative_actions(
            positions[index:index+2],quaternions[index:index+2],gripper[index:index+2],
            frame=frame,gripper_target="next",
        )[0]
    return actions,transition_valid


def action_statistics(actions: np.ndarray) -> dict:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ACTION_DIMENSION:
        raise ValueError("actions must have shape [T, 7]")
    translation_norm = np.linalg.norm(values[:, :3], axis=1) if len(values) else np.array([])
    rotation_norm = np.linalg.norm(values[:, 3:6], axis=1) if len(values) else np.array([])
    return {
        "count": int(len(values)),
        "dimension": ACTION_DIMENSION,
        "names": list(ACTION_NAMES),
        "mean": values.mean(axis=0).tolist() if len(values) else None,
        "std": values.std(axis=0).tolist() if len(values) else None,
        "min": values.min(axis=0).tolist() if len(values) else None,
        "max": values.max(axis=0).tolist() if len(values) else None,
        "p01": np.quantile(values, 0.01, axis=0).tolist() if len(values) else None,
        "p99": np.quantile(values, 0.99, axis=0).tolist() if len(values) else None,
        "translation_norm_p95_m": float(np.percentile(translation_norm, 95)) if len(values) else None,
        "translation_norm_max_m": float(translation_norm.max()) if len(values) else None,
        "rotation_norm_p95_rad": float(np.percentile(rotation_norm, 95)) if len(values) else None,
        "rotation_norm_max_rad": float(rotation_norm.max()) if len(values) else None,
        "gripper_transition_count": int(np.count_nonzero(np.diff(values[:, 6]))) if len(values) > 1 else 0,
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_npz", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    with np.load(arguments.processed_npz, allow_pickle=False) as archive:
        statistics = action_statistics(archive["actions"])
    text = json.dumps(statistics, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
