from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from processing.build_actions import build_relative_actions,build_relative_actions_masked
from processing.interpolation import (
    interpolate_linear,
    interpolate_quaternion,
    latest_discrete,
    nearest_unique_indices,
    target_timestamps,
)


def test_target_timestamps_are_uniform():
    result = target_timestamps(5, 300_000_005, 10.0)
    assert result.tolist() == [5, 100_000_005, 200_000_005, 300_000_005]


def test_nearest_unique_frame_selection():
    source = np.array([0, 31, 68, 101, 139, 201, 234, 269, 303]) * 1_000_000
    targets = np.array([0, 100, 200, 300]) * 1_000_000
    indices, errors = nearest_unique_indices(source, targets, max_error_ns=45_000_000)
    assert indices.tolist() == [0, 3, 5, 8]
    assert np.all(np.abs(errors) <= 3_000_000)


def test_linear_slerp_and_zoh():
    times = np.array([0, 100], dtype=np.int64)
    value, diagnostics = interpolate_linear(times, np.array([[0.0], [10.0]]), 50)
    assert np.allclose(value, [5.0])
    assert diagnostics.bracket_span_ns == 100
    quaternions = Rotation.from_euler("z", [[0.0], [90.0]], degrees=True).as_quat()
    quaternion, _ = interpolate_quaternion(times, quaternions, 50)
    angle = Rotation.from_quat(quaternion).as_euler("zxy", degrees=True)[0]
    assert np.isclose(angle, 45.0)
    discrete, age = latest_discrete(times, np.array([0, 1]), 75)
    assert discrete == 0 and age == 75


def test_action_uses_next_gripper_state():
    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    gripper = np.array([0, 1, 1])
    actions = build_relative_actions(positions, quaternions, gripper, frame="tool")
    assert actions.shape == (2, 7)
    assert np.allclose(actions[:, 0], 0.01)
    assert actions[:, 6].tolist() == [1.0, 1.0]


def test_invalid_observation_creates_finite_masked_action_gap():
    positions=np.array([[0.,0,0],[.01,0,0],[np.nan,0,0],[.03,0,0],[.04,0,0]])
    quaternions=np.tile([0.,0,0,1.],(5,1));gripper=np.array([0,0,-1,1,1])
    actions,valid=build_relative_actions_masked(
        positions,quaternions,gripper,np.array([True,True,False,True,True]),frame="tool"
    )
    assert valid.tolist()==[True,False,False,True]
    assert np.isfinite(actions).all()
    assert np.all(actions[~valid]==0)
    assert actions[0,0]>0 and actions[-1,0]>0
