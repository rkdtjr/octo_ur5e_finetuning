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

TOP = {"schema_version","robot","ros","cameras","sampling","camera_recording","freedrive","keyboard","gripper","replay","storage","synchronization","dataset_preprocessing","raw_topics","action_contract"}
KEYS = {
"robot":{"name","base_frame","tcp_frame","joint_names"},
"ros":{"joint_state_topic","tf_topic","tf_static_topic","controller_state_topic","trajectory_action","io_states_topic","set_io_service","robot_program_running_topic","safety_mode_topic"},
"sampling":{"demonstration_rate_hz","target_dataset_rate_hz"},
"freedrive":{"activation","auto_enable","auto_disable","controller_manager_switch_service","controller_name","motion_controller_name","enable_topic","keepalive_rate_hz","switch_timeout_sec"},
"keyboard":{"open_keys","close_keys","finish_keys","abort_keys"},
"gripper":{"semantic_open","semantic_closed","backend","output_pin","output_value_for_open","output_value_for_closed","command_on_change_only","minimum_command_interval_sec","command_timeout_sec","confirmation_timeout_sec","actuation_settle_sec","readback_from_io_states","initial_state_source"},
"replay":{"controller_joint_order","initial_joint_tolerance_rad","speed_scale","start_settle_sec","end_settle_sec","feedback_stale_sec","result_timeout_factor","result_timeout_margin_sec","max_joint_velocity_rad_s","max_joint_acceleration_rad_s2","execute_requires_program_running","execute_requires_normal_safety"},
"storage":{"output_root","rosbag_storage_id","rosbag_storage_preset_profile","video_container","raw_state_storage","compression_format","minimum_free_space_gib","overwrite"},
"synchronization":{"max_camera_time_error_ms","max_primary_wrist_difference_ms","max_pose_interpolation_gap_ms","max_joint_interpolation_gap_ms","max_gripper_age_ms","max_tcp_age_ms"},
"dataset_preprocessing":{"primary","wrist"},
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
    cr=d["camera_recording"]
    _require_keys(cr,{"capture_fps","dataset_rate_hz","encoder_queue_size","primary","wrist","preview"},"camera_recording")
    camera_keys={"enabled","source_topic","source_encoding","bayer_pattern","resolution","codec","preferred_encoder","fallback_encoder","bitrate_mbps","maxrate_mbps","bufsize_mbps","gop_size","preset","profile","pixel_format"}
    for name in ("primary","wrist"):
        _require_keys(cr[name],camera_keys,f"camera_recording.{name}")
        if len(cr[name]["resolution"])!=2 or any(int(x)<=0 for x in cr[name]["resolution"]): raise ConfigError(f"invalid {name} resolution")
        if any(cr[name][k]<=0 for k in ("bitrate_mbps","maxrate_mbps","bufsize_mbps","gop_size")): raise ConfigError(f"invalid {name} encoder setting")
    _require_keys(cr["preview"],{"enabled","fps","record"},"camera_recording.preview")
    if cr["capture_fps"]<=0 or cr["dataset_rate_hz"]<=0 or cr["encoder_queue_size"]<=0: raise ConfigError("invalid camera recording rate/queue")
    if cr["preview"]["record"]: raise ConfigError("preview recording must remain false")
    if d["storage"]["video_container"] not in ("mkv","mp4"): raise ConfigError("video_container must be mkv or mp4")
    for name in ("primary","wrist"):
        prep=d["dataset_preprocessing"][name]
        _require_keys(prep,{"output_resolution","resize_method","crop"},f"dataset_preprocessing.{name}")
        if len(prep["output_resolution"])!=2 or any(x<=0 for x in prep["output_resolution"]):raise ConfigError("invalid dataset output resolution")
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
    for section, fields in (("gripper",("minimum_command_interval_sec","command_timeout_sec","confirmation_timeout_sec","actuation_settle_sec")),("replay",("initial_joint_tolerance_rad","start_settle_sec","end_settle_sec","feedback_stale_sec","result_timeout_margin_sec"))):
        if any(d[section][k] < 0 for k in fields): raise ConfigError(f"{section} timeout/tolerance must be nonnegative")
    if d["replay"]["result_timeout_factor"] < 1: raise ConfigError("result_timeout_factor must be at least 1")
    if any(d["synchronization"][key] <= 0 for key in d["synchronization"]):
        raise ConfigError("synchronization thresholds must be positive")
    if d["freedrive"]["keepalive_rate_hz"] < 2:
        raise ConfigError("freedrive.keepalive_rate_hz must be at least 2")
    if d["freedrive"]["switch_timeout_sec"] <= 0:
        raise ConfigError("freedrive.switch_timeout_sec must be positive")
    a=d["action_contract"]
    if a["dimension"] <= 0 or len(a["normalization_mask"]) != a["dimension"]: raise ConfigError("action mask length mismatch")
    if not 0 <= a["gripper_index"] < a["dimension"]: raise ConfigError("gripper index out of range")
    return CollectorConfig(d,p.resolve())
