from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import json,os,uuid,subprocess
import yaml

STATES={"created","recording_demo","demo_recorded","demo_validated","replaying","completed","aborted","failed"}
def utc_now(): return datetime.now(timezone.utc).isoformat()
def episode_id(): return datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
def atomic_json(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(data,indent=2)+"\n"); os.replace(tmp,path)
def update_status(root,state,reason=None):
    if state not in STATES: raise ValueError("invalid episode state")
    atomic_json(Path(root)/"status.json",{"state":state,"updated_at_utc":utc_now(),"reason":reason})
def create_episode(config,instruction,episode_name=None):
    root=Path(config.storage["output_root"])/(episode_name or episode_id())
    if root.exists(): raise FileExistsError(f"episode exists: {root}")
    for p in ("demonstration","replay"): (root/p).mkdir(parents=True)
    try:
        branch=subprocess.run(["git","branch","--show-current"],capture_output=True,text=True).stdout.strip()
        commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
        dirty=bool(subprocess.run(["git","status","--porcelain"],capture_output=True,text=True).stdout)
    except OSError: branch=commit="unknown"; dirty=True
    d=config.data
    manifest={"schema_version":2,"episode_id":root.name,"instruction":instruction,"robot":d["robot"]["name"],"base_frame":d["robot"]["base_frame"],"tcp_frame":d["robot"]["tcp_frame"],"joint_names":d["robot"]["joint_names"],"camera_topics":{c["logical_name"]:c["image_topic"] for c in d["cameras"]},"gripper_semantic":{"open":0,"closed":1},"gripper_output":{"pin":d["gripper"]["output_pin"],"open":d["gripper"]["output_value_for_open"],"closed":d["gripper"]["output_value_for_closed"]},"transform_convention":"T_A_B maps p_B to p_A","created_at_utc":utc_now(),"git":{"branch":branch,"commit":commit,"dirty":dirty},"collector_version":"0.2.0","hardware_verification":[]}
    atomic_json(root/"manifest.json",manifest); update_status(root,"created")
    (root/"config_resolved.yaml").write_text(yaml.safe_dump(d,sort_keys=False))
    for f in ("demonstration/events.jsonl","replay/events.jsonl"): (root/f).touch()
    return root
