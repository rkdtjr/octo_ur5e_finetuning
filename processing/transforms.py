"""SE(3) transform helpers for UR5e Cartesian action generation."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def validate_transform(transform: np.ndarray, atol: float = 1e-7) -> None:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("invalid homogeneous bottom row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("rotation determinant must be +1")


def quaternion_pose_to_matrix(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("expected position [3] and quaternion_xyzw [4]")
    if not np.isfinite(np.concatenate([position, quaternion])).all():
        raise ValueError("pose values must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("zero quaternion is invalid")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = position
    return transform


def relative_pose_action(
    current_transform: np.ndarray,
    next_transform: np.ndarray,
    *,
    frame: str = "tool",
) -> np.ndarray:
    """Return ``[dx, dy, dz, drx, dry, drz]`` for one transition.

    ``frame='tool'`` expresses translation and rotation in the current TCP
    frame. ``frame='base'`` expresses them in the robot base frame.
    Rotation is represented as a rotation vector in radians.
    """

    validate_transform(current_transform)
    validate_transform(next_transform)
    current = np.asarray(current_transform, dtype=np.float64)
    next_value = np.asarray(next_transform, dtype=np.float64)

    if frame == "tool":
        relative_rotation = current[:3, :3].T @ next_value[:3, :3]
        translation = current[:3, :3].T @ (
            next_value[:3, 3] - current[:3, 3]
        )
    elif frame == "base":
        relative_rotation = next_value[:3, :3] @ current[:3, :3].T
        translation = next_value[:3, 3] - current[:3, 3]
    else:
        raise ValueError("frame must be 'tool' or 'base'")

    rotation_vector = Rotation.from_matrix(relative_rotation).as_rotvec()
    return np.concatenate([translation, rotation_vector])
