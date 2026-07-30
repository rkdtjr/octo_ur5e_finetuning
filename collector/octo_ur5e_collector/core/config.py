from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

class ConfigError(ValueError): pass

@dataclass(frozen=True)
class CollectorConfig:
    data: dict[str, Any]
    source: Path
    @property
    def robot(self): return self.data["robot"]
    @property
    def gripper(self): return self.data["gripper"]
    @property
    def replay(self): return self.data["replay"]
    @property
    def storage(self): return self.data["storage"]

TOP = {"schema_version","robot","ros","cameras","sampling","freedrive","keyboard","gripper","replay","storage","raw_topics","action_contract"}
KEYS = {
"robot":{"name","base_frame","tcp_frame","joint_names"},
"ros":{"joint_state_topic","tf_topic","tf_static_topic","controller_state_topic","trajectory_action","io_states_topic","set_io_service","robot_program_running_topic","safety_mode_topic"},
"sampling":{"demonstration_rate_hz","target_dataset_rate_hz"},
"freedrive":{"activation","auto_enable","auto_disable"},
"keyboard":{"open_keys","close_keys","finish_keys","abort_keys"},
"gripper":{"semantic_open","semantic_closed","backend","output_pin","output_value_for_open","output_value_for_closed","command_on_change_only","minimum_command_interval_sec","command_timeout_sec","confirmation_timeout_sec","readback_from_io_states","initial_state_source"},
"replay":{"controller_joint_order","initial_joint_tolerance_rad","speed_scale","start_settle_sec","end_settle_sec","feedback_stale_sec","result_timeout_factor","result_timeout_margin_sec","max_joint_velocity_rad_s","max_joint_acceleration_rad_s2","execute_requires_program_running","execute_requires_normal_safety"},
"storage":{"output_root","rosbag_storage_id","overwrite"},
"raw_topics":{"demonstration","replay"},
"action_contract":{"dimension","translation_unit","rotation_representation","delta_frame","gripper_index","normalization_mask"}}

def _require_keys(d, keys, where):
    missing=keys-set(d); unknown=set(d)-keys
    if missing: raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    if unknown: raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")

def load_config(path: str|Path) -> CollectorConfig:
    p=Path(path); d=yaml.safe_load(p.read_text())
    if not isinstance(d,dict): raise ConfigError("config must be a mapping")
    _require_keys(d,TOP,"root")
    if d["schema_version"] != 2: raise ConfigError("schema_version must be 2")
    for section, keys in KEYS.items():
        if not isinstance(d[section],dict): raise ConfigError(f"{section} must be a mapping")
        _require_keys(d[section],keys,section)
    cams=d["cameras"]
    if not isinstance(cams,list) or not cams: raise ConfigError("cameras must be non-empty")
    names=[]
    for i,c in enumerate(cams):
        _require_keys(c,{"logical_name","image_topic","camera_info_topic","required"},f"cameras[{i}]"); names.append(c["logical_name"])
    if len(names)!=len(set(names)): raise ConfigError("camera logical names must be unique")
    joints=d["robot"]["joint_names"]
    if len(joints)!=6 or len(set(joints))!=6: raise ConfigError("exactly 6 unique robot joints required")
    if set(d["gripper"][k] for k in ("semantic_open","semantic_closed")) != {0,1}: raise ConfigError("gripper semantics must be 0 and 1")
    if d["gripper"]["output_value_for_open"] == d["gripper"]["output_value_for_closed"]: raise ConfigError("physical gripper outputs must differ")
    if not isinstance(d["gripper"]["output_pin"],int) or not 0 <= d["gripper"]["output_pin"] <= 17: raise ConfigError("invalid output_pin")
    for k in ("demonstration_rate_hz","target_dataset_rate_hz"):
        if d["sampling"][k] <= 0: raise ConfigError(f"{k} must be positive")
    for section, fields in (("gripper",("minimum_command_interval_sec","command_timeout_sec","confirmation_timeout_sec")),("replay",("initial_joint_tolerance_rad","start_settle_sec","end_settle_sec","feedback_stale_sec","result_timeout_margin_sec"))):
        if any(d[section][k] < 0 for k in fields): raise ConfigError(f"{section} timeout/tolerance must be nonnegative")
    if d["replay"]["result_timeout_factor"] < 1: raise ConfigError("result_timeout_factor must be at least 1")
    a=d["action_contract"]
    if a["dimension"] <= 0 or len(a["normalization_mask"]) != a["dimension"]: raise ConfigError("action mask length mismatch")
    if not 0 <= a["gripper_index"] < a["dimension"]: raise ConfigError("gripper index out of range")
    return CollectorConfig(d,p.resolve())
