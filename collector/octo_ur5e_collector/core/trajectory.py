from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .transforms import ur_pose_to_matrix, validate_transform

def reorder_joints(names,values,order):
    if len(names)!=len(set(names)): raise ValueError("duplicate joint name")
    m=dict(zip(names,values))
    try: out=np.asarray([m[n] for n in order],float)
    except KeyError as e: raise ValueError(f"missing joint {e.args[0]}") from e
    if not np.isfinite(out).all(): raise ValueError("non-finite joint")
    return out

def gripper_transitions(times,states):
    t=np.asarray(times,float); s=np.asarray(states,int)
    if len(s)==0:return []
    return [{"index":int(i),"time_sec":float(t[i]),"semantic_state":int(s[i])} for i in range(1,len(s)) if s[i]!=s[i-1]]

def metrics(times,joints):
    t=np.asarray(times,float); q=np.asarray(joints,float)
    if len(t)<2:return {"max_joint_step_rad":0.0,"max_velocity_rad_s":0.0,"max_acceleration_rad_s2":0.0}
    dt=np.diff(t); dq=np.diff(q,axis=0); vel=dq/dt[:,None]
    acc=np.diff(vel,axis=0)/dt[1:,None] if len(vel)>1 else np.empty((0,q.shape[1]))
    return {"max_joint_step_rad":float(np.max(np.abs(dq))),"max_velocity_rad_s":float(np.max(np.abs(vel))),"max_acceleration_rad_s2":float(np.max(np.abs(acc))) if acc.size else 0.0}

def validate_arrays(times,joints,gripper,tcp,max_velocity=None,max_acceleration=None):
    errors=[]; t=np.asarray(times,float); q=np.asarray(joints,float); g=np.asarray(gripper); p=np.asarray(tcp,float)
    n=len(t)
    if n<2: errors.append("at least two samples required")
    if q.shape!=(n,6): errors.append("joint_position shape must be (N,6)")
    if p.shape!=(n,6): errors.append("tcp_pose6 shape must be (N,6)")
    if g.shape!=(n,): errors.append("gripper shape must be (N,)")
    if not all(np.isfinite(x).all() for x in (t,q,p)): errors.append("non-finite values")
    if n and (t[0] != 0 or np.any(np.diff(t)<=0)): errors.append("time must start at 0 and strictly increase")
    if not set(np.unique(g)).issubset({0,1}): errors.append("invalid gripper semantic")
    if p.ndim==2 and p.shape[1:]==(6,):
        for x in p:
            try: validate_transform(ur_pose_to_matrix(x))
            except ValueError as e: errors.append(f"invalid TCP: {e}"); break
    m=metrics(t,q) if q.shape==(n,6) and n>=2 and np.all(np.diff(t)>0) else {}
    if max_velocity is not None and m.get("max_velocity_rad_s",0)>max_velocity: errors.append("velocity limit exceeded")
    if max_acceleration is not None and m.get("max_acceleration_rad_s2",0)>max_acceleration: errors.append("acceleration limit exceeded")
    return {"valid":not errors,"errors":errors,"sample_count":n,"duration_sec":float(t[-1]) if n else 0.0,**m,"gripper_transitions":gripper_transitions(t,g) if g.shape==(n,) else []}

def validate_trajectory_file(path, config, require_raw_bag=True):
    path=Path(path); z=np.load(path/"demonstration/trajectory.npz")
    r=validate_arrays(z["time_from_start_sec"],z["joint_position"],z["gripper_semantic_state"],z["tcp_pose6"],config.replay["max_joint_velocity_rad_s"],config.replay["max_joint_acceleration_rad_s2"])
    raw=path/"demonstration/rosbag2"
    r["raw_bag_present"]=raw.exists() and any(raw.iterdir())
    if require_raw_bag and not r["raw_bag_present"]: r["valid"]=False; r["errors"].append("raw bag missing")
    out=path/"demonstration/validation.json"; out.write_text(json.dumps(r,indent=2)+"\n"); return r

def initial_joint_within(actual,first,tolerance):
    return bool(np.max(np.abs(np.asarray(actual)-np.asarray(first))) <= tolerance)
