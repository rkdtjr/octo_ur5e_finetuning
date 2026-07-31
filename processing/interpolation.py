"""Timestamp matching and interpolation primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class InterpolationDiagnostics:
    nearest_error_ns: int
    bracket_span_ns: int
    latest_age_ns: int


def require_strictly_increasing(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if len(array) > 1 and not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def target_timestamps(start_ns: int, end_ns: int, rate_hz: float) -> np.ndarray:
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    if end_ns < start_ns:
        raise ValueError("end_ns must not precede start_ns")
    step_ns = int(round(1e9 / float(rate_hz)))
    count = int((int(end_ns) - int(start_ns)) // step_ns) + 1
    return int(start_ns) + np.arange(count, dtype=np.int64) * step_ns


def nearest_unique_indices(
    source_ns: np.ndarray,
    target_ns: np.ndarray,
    *,
    max_error_ns: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select nearest source frames while never reusing a frame index."""

    source = require_strictly_increasing(source_ns, "source_ns")
    targets = require_strictly_increasing(target_ns, "target_ns")
    indices = np.full(len(targets), -1, dtype=np.int64)
    errors = np.full(len(targets), np.iinfo(np.int64).max, dtype=np.int64)
    previous = -1

    for target_index, target in enumerate(targets):
        insertion = int(np.searchsorted(source, target))
        candidates = {
            index
            for index in (insertion - 1, insertion)
            if previous < index < len(source)
        }
        if not candidates:
            continue
        selected = min(candidates, key=lambda index: abs(int(source[index]) - int(target)))
        error = int(source[selected]) - int(target)
        errors[target_index] = error
        if max_error_ns is not None and abs(error) > int(max_error_ns):
            continue
        indices[target_index] = selected
        previous = selected
    return indices, errors


def _bracket(times_ns: np.ndarray, target_ns: int) -> tuple[int, int, float]:
    times = require_strictly_increasing(times_ns, "times_ns")
    target = int(target_ns)
    if target < int(times[0]) or target > int(times[-1]):
        raise ValueError("target lies outside interpolation range")

    right = int(np.searchsorted(times, target, side="left"))
    if right < len(times) and int(times[right]) == target:
        return right, right, 0.0
    if right == 0 or right == len(times):
        raise ValueError("target lies outside interpolation range")
    left = right - 1
    alpha = (target - int(times[left])) / (int(times[right]) - int(times[left]))
    return left, right, float(alpha)


def _diagnostics(
    times_ns: np.ndarray,
    target_ns: int,
    left: int,
    right: int,
) -> InterpolationDiagnostics:
    target = int(target_ns)
    left_time = int(times_ns[left])
    right_time = int(times_ns[right])
    nearest = min(abs(target - left_time), abs(right_time - target))
    span = right_time - left_time
    latest_age = max(0, target - left_time)
    return InterpolationDiagnostics(nearest, span, latest_age)


def interpolate_linear(
    times_ns: np.ndarray,
    values: np.ndarray,
    target_ns: int,
) -> tuple[np.ndarray, InterpolationDiagnostics]:
    times = require_strictly_increasing(times_ns, "times_ns")
    array = np.asarray(values, dtype=np.float64)
    if len(array) != len(times):
        raise ValueError("values length does not match times")
    left, right, alpha = _bracket(times, target_ns)
    if left == right:
        value = array[left].copy()
    else:
        value = (1.0 - alpha) * array[left] + alpha * array[right]
    return value, _diagnostics(times, target_ns, left, right)


def interpolate_quaternion(
    times_ns: np.ndarray,
    quaternion_xyzw: np.ndarray,
    target_ns: int,
) -> tuple[np.ndarray, InterpolationDiagnostics]:
    times = require_strictly_increasing(times_ns, "times_ns")
    quaternions = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternions.shape != (len(times), 4):
        raise ValueError("quaternion array must have shape [N, 4]")
    left, right, alpha = _bracket(times, target_ns)
    if left == right:
        value = quaternions[left].copy()
    else:
        rotations = Rotation.from_quat(quaternions[[left, right]])
        value = Slerp([0.0, 1.0], rotations)([alpha]).as_quat()[0]
    value /= np.linalg.norm(value)
    return value, _diagnostics(times, target_ns, left, right)


def latest_discrete(
    times_ns: np.ndarray,
    values: np.ndarray,
    target_ns: int,
) -> tuple[object, int]:
    times = require_strictly_increasing(times_ns, "times_ns")
    index = int(np.searchsorted(times, int(target_ns), side="right")) - 1
    if index < 0:
        raise ValueError("no discrete value at or before target")
    return np.asarray(values)[index].item(), int(target_ns) - int(times[index])
