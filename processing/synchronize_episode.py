"""Convert one replay episode into synchronized 10 Hz transition arrays."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from .build_actions import action_statistics, build_relative_actions_masked
    from .evaluate_quality import evaluate_processed_arrays, load_thresholds
    from .interpolation import (
        interpolate_linear,
        interpolate_quaternion,
        latest_discrete,
        nearest_unique_indices,
        target_timestamps,
    )
    from .mcap_json_reader import read_json_string_topic
except ImportError:  # direct script execution
    from build_actions import action_statistics, build_relative_actions_masked
    from evaluate_quality import evaluate_processed_arrays, load_thresholds
    from interpolation import (
        interpolate_linear,
        interpolate_quaternion,
        latest_discrete,
        nearest_unique_indices,
        target_timestamps,
    )
    from mcap_json_reader import read_json_string_topic


@dataclass(frozen=True)
class CameraTimestamps:
    frame_indices: np.ndarray
    ros_stamp_ns: np.ndarray


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def source_fingerprint(episode_dir: str | Path) -> str:
    episode=Path(episode_dir);replay=episode/"replay"
    candidates=[episode/"config_resolved.yaml",episode/"manifest.json",replay/"metadata.json",replay/"robot_states.mcap",replay/"primary_timestamps.csv",replay/"wrist_timestamps.csv",replay/"primary.mkv",replay/"wrist.mkv"]
    records=[]
    for path in candidates:
        if path.exists():
            stat=path.stat();records.append((str(path.relative_to(episode)),stat.st_size,stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(records,separators=(",",":"),sort_keys=True).encode()).hexdigest()


def read_camera_timestamps(path: str | Path) -> CameraTimestamps:
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"camera timestamp CSV is empty: {path}")
    indices = np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int64)
    stamps = np.asarray([int(row["ros_stamp_ns"]) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError(f"frame_index must be contiguous from zero: {path}")
    if len(stamps) > 1 and not np.all(np.diff(stamps) > 0):
        raise ValueError(f"camera timestamps must be strictly increasing: {path}")
    return CameraTimestamps(indices, stamps)


def _source_time(state: dict, key: str) -> int:
    value = state.get(key)
    return int(state["ros_stamp_ns"] if value is None else value)


def _unique_states(states: list[dict], source_key: str) -> tuple[np.ndarray, list[dict]]:
    # Keep the latest received record for each source timestamp.
    by_time: dict[int, dict] = {}
    for state in states:
        by_time[_source_time(state, source_key)] = state
    times = np.asarray(sorted(by_time), dtype=np.int64)
    return times, [by_time[int(timestamp)] for timestamp in times]


def execution_interval(states: list[dict]) -> tuple[int, int, str]:
    executing = [state for state in states if state.get("replay_state") == "executing"]
    if len(executing) >= 2:
        return int(executing[0]["ros_stamp_ns"]), int(executing[-1]["ros_stamp_ns"]), "replay_state"

    commanded = [
        state for state in states if state.get("commanded_joint_positions") is not None
    ]
    if len(commanded) >= 2:
        return int(commanded[0]["ros_stamp_ns"]), int(commanded[-1]["ros_stamp_ns"]), "commanded_joint_positions"

    raise ValueError("cannot identify trajectory execution interval")


def build_robot_sources(states: list[dict]) -> dict[str, np.ndarray]:
    joints = [state for state in states if state.get("actual_joint_positions") is not None]
    poses = [
        state
        for state in states
        if state.get("tcp_valid", True) and state.get("actual_tcp") is not None
    ]
    grippers = [state for state in states if state.get("gripper_state") in (0, 1)]
    if len(joints) < 2 or len(poses) < 2 or not grippers:
        raise ValueError("robot state topic lacks joint, TCP, or gripper samples")

    joint_times, joint_states = _unique_states(joints, "joint_source_ros_ns")
    pose_times, pose_states = _unique_states(poses, "tcp_source_ros_ns")
    gripper_times, gripper_states = _unique_states(grippers, "gripper_source_ros_ns")

    joint_positions = np.asarray(
        [state["actual_joint_positions"] for state in joint_states], dtype=np.float64
    )
    joint_velocities = np.asarray(
        [
            state.get("actual_joint_velocities")
            if state.get("actual_joint_velocities") is not None
            else [np.nan] * 6
            for state in joint_states
        ],
        dtype=np.float64,
    )
    # Missing velocities are reconstructed by finite differences.
    if not np.isfinite(joint_velocities).all():
        seconds = joint_times.astype(np.float64) / 1e9
        joint_velocities = np.gradient(joint_positions, seconds, axis=0)

    return {
        "joint_times_ns": joint_times,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "pose_times_ns": pose_times,
        "tcp_positions": np.asarray(
            [state["actual_tcp"]["position"] for state in pose_states],
            dtype=np.float64,
        ),
        "tcp_quaternions_xyzw": np.asarray(
            [state["actual_tcp"]["quaternion_xyzw"] for state in pose_states],
            dtype=np.float64,
        ),
        "gripper_times_ns": gripper_times,
        "gripper_states": np.asarray(
            [state["gripper_state"] for state in gripper_states], dtype=np.int8
        ),
    }


def robot_match(states: list[dict], target_ns: int, thresholds: dict) -> dict | None:
    """Compatibility adapter for the collector-era single timestamp API."""
    try:
        sources=build_robot_sources(states)
        joint,joint_diag=interpolate_linear(sources["joint_times_ns"],sources["joint_positions"],target_ns)
        position,pose_diag=interpolate_linear(sources["pose_times_ns"],sources["tcp_positions"],target_ns)
        quaternion,_=interpolate_quaternion(sources["pose_times_ns"],sources["tcp_quaternions_xyzw"],target_ns)
        gripper,gripper_age=latest_discrete(sources["gripper_times_ns"],sources["gripper_states"],target_ns)
    except ValueError:return None
    pose_gap=pose_diag.bracket_span_ns/1e6;joint_gap=joint_diag.bracket_span_ns/1e6;gripper_age_ms=gripper_age/1e6
    valid=pose_gap<=thresholds["max_pose_interpolation_gap_ms"] and joint_gap<=thresholds["max_joint_interpolation_gap_ms"] and gripper_age_ms<=thresholds["max_gripper_age_ms"]
    return {"tcp_position":position.tolist(),"tcp_quaternion_xyzw":quaternion.tolist(),"joint_position":joint.tolist(),"gripper_state":int(gripper),"pose_interpolation_gap_ms":pose_gap,"joint_interpolation_gap_ms":joint_gap,"gripper_age_ms":gripper_age_ms,"robot_valid":bool(valid)}


def _match_robot(
    sources: dict[str, np.ndarray],
    targets_ns: np.ndarray,
    thresholds: dict,
) -> dict[str, np.ndarray]:
    count = len(targets_ns)
    joint_positions = np.full((count, 6), np.nan, dtype=np.float64)
    joint_velocities = np.full((count, 6), np.nan, dtype=np.float64)
    tcp_positions = np.full((count, 3), np.nan, dtype=np.float64)
    tcp_quaternions = np.full((count, 4), np.nan, dtype=np.float64)
    gripper = np.full(count, -1, dtype=np.int8)

    pose_sync = np.full(count, np.nan, dtype=np.float64)
    pose_span = np.full(count, np.nan, dtype=np.float64)
    tcp_age = np.full(count, np.nan, dtype=np.float64)
    joint_sync = np.full(count, np.nan, dtype=np.float64)
    joint_span = np.full(count, np.nan, dtype=np.float64)
    gripper_age = np.full(count, np.nan, dtype=np.float64)
    robot_valid = np.zeros(count, dtype=bool)

    sync = thresholds["synchronization"]
    for index, target in enumerate(targets_ns):
        try:
            joint_positions[index], joint_diag = interpolate_linear(
                sources["joint_times_ns"], sources["joint_positions"], int(target)
            )
            joint_velocities[index], _ = interpolate_linear(
                sources["joint_times_ns"], sources["joint_velocities"], int(target)
            )
            tcp_positions[index], pose_diag = interpolate_linear(
                sources["pose_times_ns"], sources["tcp_positions"], int(target)
            )
            tcp_quaternions[index], _ = interpolate_quaternion(
                sources["pose_times_ns"],
                sources["tcp_quaternions_xyzw"],
                int(target),
            )
            gripper_value, gripper_age_ns = latest_discrete(
                sources["gripper_times_ns"], sources["gripper_states"], int(target)
            )
            gripper[index] = int(gripper_value)
        except ValueError:
            continue

        pose_sync[index] = pose_diag.nearest_error_ns / 1e6
        pose_span[index] = pose_diag.bracket_span_ns / 1e6
        tcp_age[index] = pose_diag.latest_age_ns / 1e6
        joint_sync[index] = joint_diag.nearest_error_ns / 1e6
        joint_span[index] = joint_diag.bracket_span_ns / 1e6
        gripper_age[index] = gripper_age_ns / 1e6

        robot_valid[index] = bool(
            pose_sync[index] <= sync["pose_sync_reject_ms"]
            and pose_span[index] <= sync["pose_span_reject_ms"]
            and tcp_age[index] <= sync["tcp_age_reject_ms"]
            and joint_sync[index] <= sync["joint_sync_reject_ms"]
            and joint_span[index] <= sync["joint_span_reject_ms"]
            and gripper_age[index] <= sync["gripper_age_reject_ms"]
        )

    return {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "tcp_positions": tcp_positions,
        "tcp_quaternions_xyzw": tcp_quaternions,
        "gripper_states": gripper,
        "pose_sync_error_ms": pose_sync,
        "pose_interpolation_span_ms": pose_span,
        "tcp_source_age_ms": tcp_age,
        "joint_sync_error_ms": joint_sync,
        "joint_interpolation_span_ms": joint_span,
        "gripper_age_ms": gripper_age,
        "robot_valid_mask": robot_valid,
    }


def _camera_match(
    stamps: CameraTimestamps,
    targets_ns: np.ndarray,
    max_error_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices, errors_ns = nearest_unique_indices(
        stamps.ros_stamp_ns,
        targets_ns,
        max_error_ns=int(round(max_error_ms * 1e6)),
    )
    valid = indices >= 0
    errors_ms = errors_ns.astype(np.float64) / 1e6
    errors_ms[~valid] = np.nan
    return indices, errors_ms, valid


def _transition_view(array: np.ndarray) -> np.ndarray:
    return np.asarray(array)[:-1]


def _contiguous_valid_segments(valid_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    valid = np.asarray(valid_mask, dtype=bool)
    segment_ids = np.full(len(valid), -1, dtype=np.int32)
    segments: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate(np.r_[valid, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            segment_id = len(segments)
            segment_ids[start : end + 1] = segment_id
            segments.append(
                {
                    "segment_id": segment_id,
                    "start_step": start,
                    "end_step": end,
                    "transition_count": end - start + 1,
                }
            )
            start = None
    return segment_ids, segments


def synchronize_episode(
    episode_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    rate_hz: float | None = None,
    require_wrist: bool = False,
    action_frame: str | None = None,
    thresholds_path: str | Path | None = None,
) -> dict:
    episode = Path(episode_dir).resolve()
    replay = episode / "replay"
    if not replay.is_dir():
        raise FileNotFoundError(replay)

    config = _load_yaml(episode / "config_resolved.yaml")
    manifest = _load_json(episode / "manifest.json")
    metadata = _load_json(replay / "metadata.json")
    thresholds = load_thresholds(thresholds_path)
    target_rate = float(rate_hz or config["sampling"]["target_dataset_rate_hz"])
    action_frame = str(action_frame or config["action_contract"]["delta_frame"])

    primary = read_camera_timestamps(replay / "primary_timestamps.csv")
    wrist_path = replay / "wrist_timestamps.csv"
    wrist = read_camera_timestamps(wrist_path) if wrist_path.exists() else None
    states = read_json_string_topic(replay / metadata["robot_state_file"])
    execution_start, execution_end, interval_source = execution_interval(states)
    robot_sources = build_robot_sources(states)

    start_candidates = [
        execution_start,
        int(primary.ros_stamp_ns[0]),
        int(robot_sources["joint_times_ns"][0]),
        int(robot_sources["pose_times_ns"][0]),
        int(robot_sources["gripper_times_ns"][0]),
    ]
    end_candidates = [
        execution_end,
        int(primary.ros_stamp_ns[-1]),
        int(robot_sources["joint_times_ns"][-1]),
        int(robot_sources["pose_times_ns"][-1]),
        int(robot_sources["gripper_times_ns"][-1]),
    ]
    if wrist is not None and require_wrist:
        start_candidates.append(int(wrist.ros_stamp_ns[0]))
        end_candidates.append(int(wrist.ros_stamp_ns[-1]))
    start_ns = max(start_candidates)
    end_ns = min(end_candidates)
    # First choose approximately 10 Hz primary frames on a nominal grid. The
    # actual primary ROS header timestamps then become the observation timeline.
    # This keeps robot/action state aligned to the image rather than to an
    # artificial clock that can be tens of milliseconds away from the frame.
    nominal_targets = target_timestamps(start_ns, end_ns, target_rate)
    if len(nominal_targets) < 2:
        raise ValueError("execution interval yields fewer than two target observations")
    primary_indices, primary_error_ns = nearest_unique_indices(
        primary.ros_stamp_ns, nominal_targets, max_error_ns=None
    )
    available = primary_indices >= 0
    nominal_targets = nominal_targets[available]
    primary_indices = primary_indices[available]
    primary_error = primary_error_ns[available].astype(np.float64) / 1e6
    observation_targets = primary.ros_stamp_ns[primary_indices].astype(np.int64)
    if len(observation_targets) < 2:
        raise ValueError("fewer than two primary camera frames were selected")
    camera_limit = thresholds["synchronization"]["camera_time_error_reject_ms"]
    primary_valid = np.abs(primary_error) <= camera_limit

    if wrist is not None:
        wrist_indices, wrist_error, wrist_valid = _camera_match(
            wrist, observation_targets, camera_limit
        )
        pair_difference = np.full(len(observation_targets), np.nan, dtype=np.float64)
        paired = wrist_valid
        pair_difference[paired] = np.abs(
            observation_targets[paired]
            - wrist.ros_stamp_ns[wrist_indices[paired]]
        ) / 1e6
    else:
        wrist_indices = np.full(len(observation_targets), -1, dtype=np.int64)
        wrist_error = np.full(len(observation_targets), np.nan, dtype=np.float64)
        wrist_valid = np.zeros(len(observation_targets), dtype=bool)
        pair_difference = np.full(len(observation_targets), np.nan, dtype=np.float64)

    robot = _match_robot(robot_sources, observation_targets, thresholds)
    observation_valid = primary_valid & robot["robot_valid_mask"]
    if require_wrist:
        observation_valid &= wrist_valid

    actions,transition_valid = build_relative_actions_masked(
        robot["tcp_positions"],
        robot["tcp_quaternions_xyzw"],
        robot["gripper_states"],
        observation_valid,
        frame=action_frame,
    )
    transition_count = len(actions)
    segment_ids, valid_segments = _contiguous_valid_segments(transition_valid)
    segment_is_first = np.zeros(transition_count, dtype=bool)
    segment_is_last = np.zeros(transition_count, dtype=bool)
    for segment in valid_segments:
        segment_is_first[segment["start_step"]] = True
        segment_is_last[segment["end_step"]] = True

    arrays: dict[str, np.ndarray] = {
        "timestamps_ns": observation_targets[:-1].astype(np.int64),
        "next_timestamps_ns": observation_targets[1:].astype(np.int64),
        "time_from_start_sec": (
            (observation_targets[:-1] - observation_targets[0]) / 1e9
        ).astype(np.float64),
        "delta_time_sec": (np.diff(observation_targets) / 1e9).astype(np.float32),
        "nominal_timestamps_ns": nominal_targets[:-1].astype(np.int64),
        "primary_frame_indices": _transition_view(primary_indices).astype(np.int64),
        "wrist_frame_indices": _transition_view(wrist_indices).astype(np.int64),
        "primary_time_error_ms": _transition_view(primary_error).astype(np.float32),
        "wrist_time_error_ms": _transition_view(wrist_error).astype(np.float32),
        "primary_wrist_time_difference_ms": _transition_view(pair_difference).astype(np.float32),
        "joint_positions": _transition_view(robot["joint_positions"]).astype(np.float32),
        "joint_velocities": _transition_view(robot["joint_velocities"]).astype(np.float32),
        "tcp_positions": _transition_view(robot["tcp_positions"]).astype(np.float32),
        "tcp_quaternions_xyzw": _transition_view(robot["tcp_quaternions_xyzw"]).astype(np.float32),
        "gripper_states": _transition_view(robot["gripper_states"]).astype(np.int8),
        "actions": actions.astype(np.float32),
        "valid_mask": transition_valid.astype(bool),
        "segment_id": segment_ids.astype(np.int32),
        "segment_is_first": segment_is_first,
        "segment_is_last": segment_is_last,
        "primary_valid_mask": _transition_view(primary_valid).astype(bool),
        "wrist_valid_mask": _transition_view(wrist_valid).astype(bool),
        "robot_valid_mask": _transition_view(robot["robot_valid_mask"]).astype(bool),
        "pose_sync_error_ms": _transition_view(robot["pose_sync_error_ms"]).astype(np.float32),
        "pose_interpolation_span_ms": _transition_view(robot["pose_interpolation_span_ms"]).astype(np.float32),
        "joint_sync_error_ms": _transition_view(robot["joint_sync_error_ms"]).astype(np.float32),
        "joint_interpolation_span_ms": _transition_view(robot["joint_interpolation_span_ms"]).astype(np.float32),
        "gripper_age_ms": _transition_view(robot["gripper_age_ms"]).astype(np.float32),
        "tcp_source_age_ms": _transition_view(robot["tcp_source_age_ms"]).astype(np.float32),
        "is_first": segment_is_first.copy(),
        "is_last": segment_is_last.copy(),
        "is_terminal": segment_is_last.copy(),
    }

    output = Path(output_dir).resolve() if output_dir else episode / "processed"
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "synchronized_episode.npz"
    np.savez_compressed(npz_path, **arrays)

    report = evaluate_processed_arrays(
        arrays, thresholds, require_wrist=require_wrist
    )
    statistics = action_statistics(actions[transition_valid])
    report.update(
        {
            "episode_id": manifest["episode_id"],
            "instruction": manifest["instruction"],
            "rate_hz": target_rate,
            "execution_interval_source": interval_source,
            "execution_start_ros_ns": int(execution_start),
            "execution_end_ros_ns": int(execution_end),
            "processed_start_ros_ns": int(observation_targets[0]),
            "processed_end_ros_ns": int(observation_targets[-1]),
            "effective_rate_hz": float((len(observation_targets) - 1) / ((observation_targets[-1] - observation_targets[0]) / 1e9)),
            "delta_time_p95_ms": float(np.percentile(np.diff(observation_targets) / 1e6, 95)),
            "delta_time_max_ms": float(np.max(np.diff(observation_targets) / 1e6)),
            "action_statistics": statistics,
            "gripper_transition_count": statistics["gripper_transition_count"],
            "valid_segment_count": len(valid_segments),
            "valid_segments": valid_segments,
        }
    )
    (output / "quality_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "action_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )

    camera_metadata=metadata.get("cameras",{})
    primary_metadata=camera_metadata.get("primary")
    if not primary_metadata:raise ValueError("replay metadata lacks primary camera")
    wrist_metadata=camera_metadata.get("wrist")
    processing_manifest = {
        "schema_version": 1,
        "source_fingerprint": source_fingerprint(episode),
        "episode_id": manifest["episode_id"],
        "source_episode": str(episode),
        "instruction": manifest["instruction"],
        "robot": manifest["robot"],
        "rate_hz": target_rate,
        "transition_count": int(transition_count),
        "valid_transition_count": int(transition_valid.sum()),
        "valid_segment_count": len(valid_segments),
        "valid_segments": valid_segments,
        "require_wrist": bool(require_wrist),
        "action": {
            "dimension": 7,
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            "delta_frame": action_frame,
            "translation_unit": "meter",
            "rotation_representation": "rotation_vector",
            "gripper_semantic": "absolute next-state, 0=open, 1=closed",
            "normalization_mask": [True, True, True, True, True, True, False],
        },
        "camera_frame_selection": "nearest unique primary frame to nominal 10 Hz grid; selected primary ROS stamps become the actual observation timeline",
        "robot_interpolation": {
            "joint_position": "linear",
            "joint_velocity": "linear",
            "tcp_position": "linear",
            "tcp_orientation": "SLERP quaternion_xyzw",
            "gripper": "zero-order hold",
        },
        "source_files": {
            "robot_states": str(replay / metadata["robot_state_file"]),
            "primary_video": str(replay / primary_metadata["file"]),
            "primary_timestamps": str(replay / primary_metadata["timestamps"]),
            "wrist_video": str(replay / wrist_metadata["file"]) if wrist_metadata else None,
            "wrist_timestamps": str(replay / wrist_metadata["timestamps"]) if wrist_metadata else None,
        },
        "outputs": {
            "npz": npz_path.name,
            "quality_report": "quality_report.json",
            "action_statistics": "action_statistics.json",
            "synchronization_index": "synchronization_index.csv",
        },
    }
    (output / "processing_manifest.yaml").write_text(
        yaml.safe_dump(processing_manifest, sort_keys=False), encoding="utf-8"
    )

    with (output / "synchronization_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "step",
            "target_ros_ns",
            "next_target_ros_ns",
            "primary_frame_index",
            "wrist_frame_index",
            "primary_time_error_ms",
            "wrist_time_error_ms",
            "primary_wrist_difference_ms",
            "pose_sync_error_ms",
            "pose_span_ms",
            "joint_sync_error_ms",
            "joint_span_ms",
            "gripper_age_ms",
            "tcp_source_age_ms",
            "gripper_state",
            "action_gripper",
            "valid",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(transition_count):
            writer.writerow(
                {
                    "step": index,
                    "target_ros_ns": int(arrays["timestamps_ns"][index]),
                    "next_target_ros_ns": int(arrays["next_timestamps_ns"][index]),
                    "primary_frame_index": int(arrays["primary_frame_indices"][index]),
                    "wrist_frame_index": int(arrays["wrist_frame_indices"][index]),
                    "primary_time_error_ms": float(arrays["primary_time_error_ms"][index]),
                    "wrist_time_error_ms": float(arrays["wrist_time_error_ms"][index]),
                    "primary_wrist_difference_ms": float(arrays["primary_wrist_time_difference_ms"][index]),
                    "pose_sync_error_ms": float(arrays["pose_sync_error_ms"][index]),
                    "pose_span_ms": float(arrays["pose_interpolation_span_ms"][index]),
                    "joint_sync_error_ms": float(arrays["joint_sync_error_ms"][index]),
                    "joint_span_ms": float(arrays["joint_interpolation_span_ms"][index]),
                    "gripper_age_ms": float(arrays["gripper_age_ms"][index]),
                    "tcp_source_age_ms": float(arrays["tcp_source_age_ms"][index]),
                    "gripper_state": int(arrays["gripper_states"][index]),
                    "action_gripper": float(arrays["actions"][index, 6]),
                    "valid": bool(arrays["valid_mask"][index]),
                }
            )

    return {
        "episode_id": manifest["episode_id"],
        "output_dir": str(output),
        "processed_npz": str(npz_path),
        "transition_count": int(transition_count),
        "valid_transition_count": int(transition_valid.sum()),
        "valid_ratio": float(transition_valid.mean()),
        "quality": report["overall"],
        "quality_grade": report["quality_grade"],
        "verdict": report["verdict"],
        "warnings": report["warnings"],
        "hard_failures": report["hard_failures"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rate-hz", type=float)
    parser.add_argument("--require-wrist", action="store_true")
    parser.add_argument("--action-frame", choices=("tool", "base"))
    parser.add_argument("--thresholds", type=Path)
    arguments = parser.parse_args()
    result = synchronize_episode(
        arguments.episode_dir,
        output_dir=arguments.output_dir,
        rate_hz=arguments.rate_hz,
        require_wrist=arguments.require_wrist,
        action_frame=arguments.action_frame,
        thresholds_path=arguments.thresholds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
