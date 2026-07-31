"""Evaluate processed transitions while separating warnings from hard rejects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_DEFAULT_THRESHOLDS = Path(__file__).with_name("quality_thresholds.yaml")


def load_thresholds(path: str | Path | None = None) -> dict:
    with Path(path or _DEFAULT_THRESHOLDS).open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("quality thresholds must be a mapping")
    return result


def _finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    return result[np.isfinite(result)]


def _percentile(values: np.ndarray, q: float) -> float | None:
    values = _finite(values)
    return float(np.percentile(values, q)) if len(values) else None


def _maximum(values: np.ndarray) -> float | None:
    values = _finite(values)
    return float(np.max(values)) if len(values) else None


def _rated_upper(
    name: str,
    value: float | None,
    warning_max: float,
    reject_max: float,
    unit: str,
    *,
    reject_enabled: bool = True,
) -> dict[str, Any]:
    if value is None:
        rating = "REJECT"
    elif value <= warning_max:
        rating = "PASS"
    elif value <= reject_max or not reject_enabled:
        rating = "WARN"
    else:
        rating = "REJECT"
    return {
        "name": name,
        "rating": rating,
        "value": value,
        "unit": unit,
        "warning_max": float(warning_max),
        "reject_max": float(reject_max),
    }


def evaluate_processed_arrays(
    data: dict[str, np.ndarray],
    thresholds: dict,
    *,
    require_wrist: bool = False,
) -> dict:
    required = {
        "timestamps_ns",
        "next_timestamps_ns",
        "actions",
        "valid_mask",
        "primary_time_error_ms",
        "pose_sync_error_ms",
        "pose_interpolation_span_ms",
        "joint_sync_error_ms",
        "joint_interpolation_span_ms",
        "gripper_age_ms",
        "tcp_source_age_ms",
    }
    if require_wrist:
        required.update({"wrist_time_error_ms", "primary_wrist_time_difference_ms"})
    missing = sorted(required - set(data))
    if missing:
        return {
            "schema_version": 1,
            "overall": "REJECT",
            "verdict": "not_ready",
            "hard_failures": ["missing_arrays:" + ",".join(missing)],
            "warnings": [],
            "checks": [],
        }

    timestamps = np.asarray(data["timestamps_ns"], dtype=np.int64)
    next_timestamps = np.asarray(data["next_timestamps_ns"], dtype=np.int64)
    actions = np.asarray(data["actions"], dtype=np.float64)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    hard_failures: list[str] = []

    if not len(timestamps):
        hard_failures.append("empty_episode")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        hard_failures.append("non_monotonic_timestamps")
    if timestamps.shape != next_timestamps.shape or np.any(next_timestamps <= timestamps):
        hard_failures.append("invalid_transition_timestamps")
    if actions.shape != (len(timestamps), 7):
        hard_failures.append("invalid_action_shape")
    if valid.shape != (len(timestamps),):
        hard_failures.append("invalid_valid_mask_shape")
    if not np.isfinite(actions).all():
        hard_failures.append("non_finite_action")

    synchronization = thresholds["synchronization"]
    limits = thresholds["actions"]
    minimum = thresholds["minimum"]
    checks = [
        _rated_upper(
            "primary_time_error_p95",
            _percentile(np.abs(data["primary_time_error_ms"]), 95),
            synchronization["camera_time_error_warning_ms"],
            synchronization["camera_time_error_reject_ms"],
            "ms",
        ),
        _rated_upper(
            "pose_sync_error_p95",
            _percentile(data["pose_sync_error_ms"], 95),
            synchronization["pose_sync_warning_ms"],
            synchronization["pose_sync_reject_ms"],
            "ms",
        ),
        _rated_upper(
            "pose_interpolation_span_p95",
            _percentile(data["pose_interpolation_span_ms"], 95),
            synchronization["pose_span_warning_ms"],
            synchronization["pose_span_reject_ms"],
            "ms",
        ),
        _rated_upper(
            "joint_sync_error_p95",
            _percentile(data["joint_sync_error_ms"], 95),
            synchronization["joint_sync_warning_ms"],
            synchronization["joint_sync_reject_ms"],
            "ms",
        ),
        _rated_upper(
            "joint_interpolation_span_p95",
            _percentile(data["joint_interpolation_span_ms"], 95),
            synchronization["joint_span_warning_ms"],
            synchronization["joint_span_reject_ms"],
            "ms",
        ),
        _rated_upper(
            "gripper_age_p95",
            _percentile(data["gripper_age_ms"], 95),
            synchronization["gripper_age_warning_ms"],
            synchronization["gripper_age_reject_ms"],
            "ms",
        ),
        _rated_upper(
            "tcp_source_age_p95",
            _percentile(data["tcp_source_age_ms"], 95),
            synchronization["tcp_age_warning_ms"],
            synchronization["tcp_age_reject_ms"],
            "ms",
        ),
        _rated_upper(
            "translation_action_norm_max",
            _maximum(np.linalg.norm(actions[:, :3], axis=1)) if len(actions) else None,
            limits["translation_norm_warning_m"],
            limits["translation_norm_reject_m"],
            "m/step",
        ),
        _rated_upper(
            "rotation_action_norm_max",
            _maximum(np.linalg.norm(actions[:, 3:6], axis=1)) if len(actions) else None,
            limits["rotation_norm_warning_rad"],
            limits["rotation_norm_reject_rad"],
            "rad/step",
        ),
    ]

    wrist_available = "wrist_time_error_ms" in data
    if wrist_available:
        checks.append(
            _rated_upper(
                "wrist_time_error_p95",
                _percentile(np.abs(data["wrist_time_error_ms"]), 95),
                synchronization["camera_time_error_warning_ms"],
                synchronization["camera_time_error_reject_ms"],
                "ms",
                reject_enabled=require_wrist,
            )
        )
    if "primary_wrist_time_difference_ms" in data:
        checks.append(
            _rated_upper(
                "primary_wrist_difference_p95",
                _percentile(data["primary_wrist_time_difference_ms"], 95),
                synchronization["camera_pair_warning_ms"],
                synchronization["camera_pair_reject_ms"],
                "ms",
                reject_enabled=require_wrist,
            )
        )

    valid_ratio = float(valid.mean()) if len(valid) else 0.0
    if len(timestamps) < int(minimum["transition_count"]):
        hard_failures.append("too_few_transitions")
    if valid_ratio < float(minimum["valid_ratio_reject"]):
        hard_failures.append("valid_ratio_below_reject_threshold")

    warnings = [item["name"] for item in checks if item["rating"] == "WARN"]
    hard_failures.extend(item["name"] for item in checks if item["rating"] == "REJECT")
    if hard_failures:
        overall, verdict = "REJECT", "not_ready"
    elif warnings or valid_ratio < float(minimum["valid_ratio_warning"]):
        overall, verdict = "WARN_ACCEPTED", "usable_with_warning"
        if valid_ratio < float(minimum["valid_ratio_warning"]):
            warnings.append("valid_ratio_below_warning_threshold")
    else:
        overall, verdict = "ACCEPTED", "ready"

    return {
        "schema_version": 1,
        "overall": overall,
        "verdict": verdict,
        "require_wrist": bool(require_wrist),
        "transition_count": int(len(timestamps)),
        "valid_count": int(valid.sum()),
        "valid_ratio": valid_ratio,
        "hard_failures": sorted(set(hard_failures)),
        "warnings": sorted(set(warnings)),
        "checks": checks,
    }


def evaluate_processed_npz(
    path: str | Path,
    thresholds_path: str | Path | None = None,
    *,
    require_wrist: bool = False,
) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    return evaluate_processed_arrays(
        data, load_thresholds(thresholds_path), require_wrist=require_wrist
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_npz", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--require-wrist", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate_processed_npz(
        arguments.processed_npz,
        arguments.thresholds,
        require_wrist=arguments.require_wrist,
    )
    text = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
