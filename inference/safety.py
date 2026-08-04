"""Pure safety transforms for receding-horizon UR5e policy execution."""
from __future__ import annotations
import numpy as np


DEFAULT_WORKSPACE_MIN=np.array([-1.49,-1.11,-1.12],float)
DEFAULT_WORKSPACE_MAX=np.array([1.64,1.22,1.54],float)


def limit_action(action,max_translation_m=.003,max_rotation_rad=.02):
    value=np.asarray(action,float).copy()
    if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError("policy action must be finite shape (7,)")
    for section,limit in ((slice(0,3),max_translation_m),(slice(3,6),max_rotation_rad)):
        norm=float(np.linalg.norm(value[section]))
        if norm>limit:value[section]*=limit/norm
    value[6]=float(np.clip(value[6],0,1));return value


def validate_target_position(position,minimum=DEFAULT_WORKSPACE_MIN,maximum=DEFAULT_WORKSPACE_MAX):
    point=np.asarray(position,float);low=np.asarray(minimum,float);high=np.asarray(maximum,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError("target position must be finite shape (3,)")
    if np.any(point<low) or np.any(point>high):raise ValueError(f"target outside workspace: {point.tolist()}")
    return point


def gripper_transition(value,current,close_threshold=.7,open_threshold=.3):
    value=float(value)
    if current is None:return 1 if value>=close_threshold else 0
    if current==0 and value>=close_threshold:return 1
    if current==1 and value<=open_threshold:return 0
    return current


class GripperDebouncer:
    """Stateful hysteresis requiring consecutive evidence before switching."""
    def __init__(self,close_threshold=.9,open_threshold=.1,close_steps=3,open_steps=3):
        if not 0<=open_threshold<close_threshold<=1:raise ValueError("gripper thresholds must satisfy 0 <= open < close <= 1")
        if close_steps<1 or open_steps<1:raise ValueError("gripper debounce steps must be positive")
        self.close_threshold=float(close_threshold);self.open_threshold=float(open_threshold)
        self.close_steps=int(close_steps);self.open_steps=int(open_steps);self._high=0;self._low=0

    def update(self,value,current):
        value=float(value)
        if not np.isfinite(value):raise ValueError("gripper value must be finite")
        state=0 if current is None else int(current)
        if state==0:
            self._low=0;self._high=self._high+1 if value>=self.close_threshold else 0
            if self._high>=self.close_steps:self._high=0;return 1
        elif state==1:
            self._high=0;self._low=self._low+1 if value<=self.open_threshold else 0
            if self._low>=self.open_steps:self._low=0;return 0
        else:raise ValueError("current gripper state must be 0, 1, or None")
        return state
