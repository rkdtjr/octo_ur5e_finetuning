from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import os,shutil,subprocess

@dataclass
class Check:
    name:str; ok:bool; detail:str; required:bool=True

def _lines(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=10)
    return p.returncode,set(p.stdout.splitlines()),p.stderr.strip()

def _topic_once(topic,field):
    try:
        p=subprocess.run(["ros2","topic","echo","--once","--field",field,topic],capture_output=True,text=True,timeout=3)
        if p.returncode != 0:
            return False,p.stderr.strip() or p.stdout.strip()
        # `ros2 topic echo --once` terminates YAML documents with `---`.
        # Return only the scalar field value so state comparisons do not
        # accidentally include the document separator.
        values=[line.strip() for line in p.stdout.splitlines() if line.strip() and line.strip()!="---"]
        return bool(values),values[0] if values else "empty message"
    except subprocess.TimeoutExpired:
        return False,"no message within 3 seconds"

def run_preflight(config,execute=False,replay=False,freedrive=False):
    d=config.data; checks=[]
    checks.append(Check("ros2_cli",shutil.which("ros2") is not None,shutil.which("ros2") or "not found"))
    root=Path(d["storage"]["output_root"])
    try: root.mkdir(parents=True,exist_ok=True); ok=os.access(root,os.W_OK)
    except OSError as e: ok=False
    checks.append(Check("output_writable",ok,str(root)))
    free_gib=shutil.disk_usage(root).free/1024**3 if root.exists() else 0
    minimum=d["storage"]["minimum_free_space_gib"]
    checks.append(Check("disk_free_space",free_gib>=minimum,f"{free_gib:.1f} GiB free, minimum {minimum:.1f} GiB",replay))
    if not checks[0].ok:return checks
    rc,topics,err=_lines(["ros2","topic","list"])
    checks.append(Check("ros_graph",rc==0,err or f"{len(topics)} topics"))
    required_topics=[d["ros"]["joint_state_topic"],d["ros"]["tf_topic"],d["ros"]["io_states_topic"]]
    required_topics += [c["image_topic"] for c in d["cameras"] if c["required"] and replay]
    for topic in required_topics: checks.append(Check(f"topic:{topic}",topic in topics,"present" if topic in topics else "missing"))
    program_topic=d["ros"]["robot_program_running_topic"]
    safety_topic=d["ros"]["safety_mode_topic"]
    if program_topic in topics:
        ok,value=_topic_once(program_topic,"data")
        running=ok and value.lower()=="true"
        checks.append(Check("robot_program_running",running,value,(execute or freedrive) and d["replay"]["execute_requires_program_running"]))
    else: checks.append(Check("robot_program_running",False,"topic missing",(execute or freedrive) and d["replay"]["execute_requires_program_running"]))
    if safety_topic in topics:
        ok,value=_topic_once(safety_topic,"mode")
        normal=ok and value.strip()=="1"
        checks.append(Check("safety_mode_normal",normal,value,(execute or freedrive) and d["replay"]["execute_requires_normal_safety"]))
    else: checks.append(Check("safety_mode_normal",False,"topic missing",(execute or freedrive) and d["replay"]["execute_requires_normal_safety"]))
    rc,services,err=_lines(["ros2","service","list"])
    svc=d["ros"]["set_io_service"]; checks.append(Check(f"service:{svc}",svc in services,"present" if svc in services else "missing",execute))
    if freedrive:
        switch=d["freedrive"]["controller_manager_switch_service"]
        checks.append(Check(f"service:{switch}",switch in services,"present" if switch in services else "missing",True))
    rc,actions,err=_lines(["ros2","action","list"])
    action=d["ros"]["trajectory_action"]; checks.append(Check(f"action:{action}",action in actions,"present" if action in actions else "missing",execute and replay))
    storage=d["storage"]["rosbag_storage_id"]
    p=subprocess.run(["ros2","bag","record","--help"],capture_output=True,text=True)
    checks.append(Check(f"rosbag_storage:{storage}",p.returncode==0,storage+" requested"))
    if replay:
        from ..core.video_recording import select_encoder
        for name in ("primary","wrist"):
            camera=d["camera_recording"][name]
            if not camera["enabled"]:continue
            try:
                selected=select_encoder(camera["preferred_encoder"],camera["fallback_encoder"])
                checks.append(Check(f"encoder:{name}",True,selected))
            except RuntimeError as e:checks.append(Check(f"encoder:{name}",False,str(e)))
    return checks

def preflight_ok(checks): return all(c.ok for c in checks if c.required)
def as_json(checks): return [asdict(c) for c in checks]
