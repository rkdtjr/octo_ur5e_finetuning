from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from processing.batch_process import discover_episodes
from processing.synchronize_episode import source_fingerprint,synchronize_episode
from .ur5e_pick.episodes import load_trajectory,plan_dataset

def _grades(value):
    result=tuple(x.strip().upper() for x in value.split(",") if x.strip())
    if not result or not set(result)<={"GOOD","WARNING","BAD"}:raise ValueError("grades must be GOOD, WARNING, and/or BAD")
    return result

def doctor(root):
    import yaml
    episodes=discover_episodes(Path(root));current=0;stale=0
    for episode in episodes:
        manifest=episode/"processed/processing_manifest.yaml"
        if not manifest.exists():continue
        value=yaml.safe_load(manifest.read_text()) or {}
        if value.get("source_fingerprint")==source_fingerprint(episode):current+=1
        else:stale+=1
    dependencies={}
    for name in ("cv2","tensorflow","tensorflow_datasets"):
        try:
            module=__import__(name)
            if name=="tensorflow_datasets":getattr(module,"core")
            dependencies[name]="OK"
        except Exception as error:dependencies[name]=f"ERROR: {type(error).__name__}: {error}"
    return {"raw_episode_count":len(episodes),"current_processed_count":current,"stale_processed_count":stale,"pending_episode_count":len(episodes)-current-stale,"dependencies":dependencies,"can_process":dependencies["cv2"]=="OK","can_build_rlds":all(dependencies[x]=="OK" for x in ("tensorflow","tensorflow_datasets"))}

def process_pending(root,require_wrist=False,force=False):
    import yaml
    results=[]
    for episode in discover_episodes(Path(root)):
        target=episode/"processed/synchronized_episode.npz";manifest=episode/"processed/processing_manifest.yaml";current=False
        if target.exists() and manifest.exists():current=(yaml.safe_load(manifest.read_text()) or {}).get("source_fingerprint")==source_fingerprint(episode)
        if current and not force:results.append({"episode_id":episode.name,"status":"reused"});continue
        result=synchronize_episode(episode,require_wrist=require_wrist);results.append({"episode_id":episode.name,"status":"processed",**result})
    return results

def main(argv=None):
    parser=argparse.ArgumentParser(prog="octo-dataset",description="Incremental processing and safe RLDS/TFDS build pipeline")
    sub=parser.add_subparsers(dest="command",required=True)
    d=sub.add_parser("doctor");d.add_argument("root",nargs="?",default="data/raw")
    c=sub.add_parser("process");c.add_argument("root",nargs="?",default="data/raw");c.add_argument("--require-wrist",action="store_true");c.add_argument("--force",action="store_true");c.add_argument("--execute",action="store_true")
    p=sub.add_parser("plan");p.add_argument("root",nargs="?",default="data/raw");p.add_argument("--include-grades",default="GOOD,WARNING");p.add_argument("--require-wrist",action="store_true");p.add_argument("--val-episodes",type=int,default=3);p.add_argument("--output")
    b=sub.add_parser("build");b.add_argument("root",nargs="?",default="data/raw");b.add_argument("--output",default="data/rlds");b.add_argument("--include-grades",default="GOOD,WARNING");b.add_argument("--require-wrist",action="store_true");b.add_argument("--val-episodes",type=int,default=3);b.add_argument("--force-process",action="store_true");b.add_argument("--execute",action="store_true")
    a=parser.parse_args(argv)
    try:
        if a.command=="doctor":result=doctor(a.root)
        elif a.command=="process":
            preview=doctor(a.root)
            if not a.execute:result={"mode":"DRY_RUN","would_process":preview["pending_episode_count"]+preview["stale_processed_count"],"would_reuse":preview["current_processed_count"],"next":"repeat with --execute"}
            else:result={"mode":"EXECUTE","episodes":process_pending(a.root,a.require_wrist,a.force)}
        elif a.command=="plan":
            result=plan_dataset(a.root,_grades(a.include_grades),a.require_wrist,a.val_episodes)
            if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
        else:
            preview=doctor(a.root)
            if not a.execute:
                result={"mode":"DRY_RUN","would_process":preview["pending_episode_count"]+preview["stale_processed_count"],"would_reuse":preview["current_processed_count"],"output":str(Path(a.output).resolve()),"include_grades":list(_grades(a.include_grades)),"require_wrist":a.require_wrist,"can_build_rlds":preview["can_build_rlds"],"next":"repeat with --execute"}
            else:
                if not preview["can_build_rlds"]:raise RuntimeError("RLDS build dependencies missing; install with: pip install -e '.[rlds]'")
                processing=process_pending(a.root,a.require_wrist,a.force_process);plan=plan_dataset(a.root,_grades(a.include_grades),a.require_wrist,a.val_episodes)
                if not plan["trajectory_count"]:raise RuntimeError("no eligible valid trajectories")
                # Decode one complete trajectory before starting TFDS writes.
                key,sample=load_trajectory(plan["trajectories"][0])
                if len(sample["steps"])<1:raise RuntimeError("validation trajectory is empty")
                from .ur5e_pick.tfds_builder import Ur5ePick
                output=Path(a.output)
                if output.exists() and any(output.iterdir()):raise RuntimeError(f"output is not empty: {output}")
                builder=Ur5ePick(data_dir=str(output),plan=plan);builder.download_and_prepare()
                manifest={"schema_version":1,"dataset":"ur5e_pick","builder_version":str(builder.VERSION),"plan":plan,"processing":processing,"validation_example":key,"data_dir":str(Path(builder.data_dir).resolve())}
                output.mkdir(parents=True,exist_ok=True);(output/"build_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
                result={"mode":"EXECUTE","dataset_dir":builder.data_dir,"trajectory_count":plan["trajectory_count"],"train_count":plan["train_count"],"val_count":plan["val_count"]}
        print(json.dumps(result,indent=2))
    except (ValueError,RuntimeError,FileNotFoundError) as e:
        print(f"error: {e}",file=sys.stderr);raise SystemExit(2)

if __name__=="__main__":main()
