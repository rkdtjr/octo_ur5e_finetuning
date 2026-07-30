from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from .core.config import load_config,ConfigError
from .core.episode import create_episode
from .core.trajectory import validate_trajectory_file

DEFAULT_CONFIG="collector/config/collector.yaml"
def parser():
    p=argparse.ArgumentParser(prog="octo-collector",description="Safe UR5e demonstration collector (hardware actions require --execute)")
    p.add_argument("--version",action="version",version="%(prog)s 0.2.0")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("doctor",help="read-only ROS/config preflight"); d.add_argument("--config",default=DEFAULT_CONFIG); d.add_argument("--replay",action="store_true")
    r=sub.add_parser("record-demo",help="record a freedrive demonstration"); r.add_argument("--config",default=DEFAULT_CONFIG); r.add_argument("--instruction",required=True); r.add_argument("--execute",action="store_true"); r.add_argument("--initial-gripper",choices=["open","closed"])
    v=sub.add_parser("validate-demo",help="validate structured trajectory"); v.add_argument("episode"); v.add_argument("--config",default=DEFAULT_CONFIG); v.add_argument("--allow-missing-bag",action="store_true")
    x=sub.add_parser("replay",help="validate/dry-run or explicitly execute replay"); x.add_argument("episode"); x.add_argument("--config",default=DEFAULT_CONFIG); x.add_argument("--execute",action="store_true"); x.add_argument("--wall-clock-gripper-fallback",action="store_true")
    i=sub.add_parser("inspect",help="show episode metadata"); i.add_argument("episode")
    return p

def _doctor(cfg,execute=False,replay=False):
    from .ros_adapters.preflight import run_preflight,preflight_ok
    checks=run_preflight(cfg,execute,replay)
    for c in checks: print(f"{'OK' if c.ok else 'FAIL'} {'required' if c.required else 'optional'} {c.name}: {c.detail}")
    return 0 if preflight_ok(checks) else 2

def _inspect(path):
    p=Path(path)
    for name in ("manifest.json","status.json","demonstration/validation.json","replay/metadata.json","replay/quality_report.json","replay/execution_summary.json"):
        f=p/name
        if f.exists(): print(f"\n{name}\n{json.dumps(json.loads(f.read_text()),indent=2)}")
    t=p/"demonstration/trajectory.npz"
    if t.exists():
        z=np.load(t); times=z["time_from_start_sec"]; g=z["gripper_semantic_state"]
        print(f"\ntrajectory: samples={len(times)} duration={times[-1] if len(times) else 0:.3f}s gripper_transitions={int(np.sum(np.diff(g)!=0))}")
    return 0

def main(argv=None) -> None:
    a=parser().parse_args(argv)
    try:
        if a.command=="inspect": code=_inspect(a.episode)
        else:
            cfg=load_config(a.config)
            if a.command=="doctor": code=_doctor(cfg,replay=a.replay)
            elif a.command=="validate-demo":
                result=validate_trajectory_file(a.episode,cfg,not a.allow_missing_bag); print(json.dumps(result,indent=2)); code=0 if result["valid"] else 2
            elif a.command=="record-demo":
                from .record_demonstration_node import run_recording
                code=run_recording(cfg,a.instruction,a.execute,a.initial_gripper)
            elif a.command=="replay":
                from .replay_trajectory_node import run_replay
                code=run_replay(Path(a.episode),cfg,a.execute,a.wall_clock_gripper_fallback)
    except (ConfigError,ValueError,FileNotFoundError,RuntimeError) as e:
        print(f"error: {e}",file=sys.stderr); code=2
    raise SystemExit(code)

if __name__ == "__main__":
    main()
