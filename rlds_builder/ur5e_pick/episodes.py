from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2,numpy as np,yaml
from processing.batch_process import discover_episodes
from processing.convert_to_rlds import decode_selected_rgb
from processing.synchronize_episode import source_fingerprint

def _json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def _yaml(path):return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
def center_crop_resize(rgb,size):
    height,width=map(int,size);h,w=rgb.shape[:2];side=min(h,w);y=(h-side)//2;x=(w-side)//2
    return cv2.resize(rgb[y:y+side,x:x+side],(width,height),interpolation=cv2.INTER_AREA)

def plan_dataset(root,include_grades=("GOOD","WARNING"),require_wrist=False,val_episode_count=3):
    entries=[];skipped=[]
    for episode in discover_episodes(Path(root)):
        processed=episode/"processed";quality_path=processed/"quality_report.json";npz_path=processed/"synchronized_episode.npz"
        processing_manifest=processed/"processing_manifest.yaml"
        if not quality_path.exists() or not npz_path.exists() or not processing_manifest.exists():skipped.append({"episode_id":episode.name,"reason":"not_processed"});continue
        if (_yaml(processing_manifest) or {}).get("source_fingerprint")!=source_fingerprint(episode):skipped.append({"episode_id":episode.name,"reason":"stale_processed"});continue
        quality=_json(quality_path);grade=quality.get("quality_grade",{"ACCEPTED":"GOOD","WARN_ACCEPTED":"WARNING","REJECT":"BAD"}.get(quality.get("overall"),"BAD"))
        if grade not in include_grades:skipped.append({"episode_id":episode.name,"reason":f"quality_{grade.lower()}"});continue
        with np.load(npz_path,allow_pickle=False) as data:
            segment_ids=data["segment_id"]
            segments=sorted(int(x) for x in np.unique(segment_ids) if int(x)>=0)
            usable_segments=[]
            for segment in segments:
                step_count=int(np.count_nonzero(segment_ids==segment))
                if step_count<2:
                    skipped.append({"episode_id":episode.name,"segment_id":segment,"reason":"fewer_than_2_steps"})
                else:
                    usable_segments.append(segment)
            segments=usable_segments
            wrist_ok=bool(np.all(data["wrist_frame_indices"][data["valid_mask"]]>=0))
        if require_wrist and not wrist_ok:skipped.append({"episode_id":episode.name,"reason":"wrist_required"});continue
        entries.extend({"episode_id":episode.name,"episode_path":str(episode.resolve()),"segment_id":segment,"quality_grade":grade} for segment in segments)
    entries.sort(key=lambda x:(x["episode_id"],x["segment_id"]))
    episode_ids=sorted({x["episode_id"] for x in entries})
    requested=max(0,int(val_episode_count))
    validation_size=min(requested,max(0,len(episode_ids)-1))
    ranked=sorted(episode_ids,key=lambda value:hashlib.sha256(value.encode()).hexdigest())
    val_episode_ids=set(ranked[:validation_size])
    for entry in entries:entry["split"]="val" if entry["episode_id"] in val_episode_ids else "train"
    return {"schema_version":1,"source_root":str(Path(root).resolve()),"include_grades":list(include_grades),"require_wrist":require_wrist,"val_episode_count":validation_size,"val_episode_ids":sorted(val_episode_ids),"trajectory_count":len(entries),"train_count":sum(x["split"]=="train" for x in entries),"val_count":sum(x["split"]=="val" for x in entries),"skipped":skipped,"trajectories":entries}

def load_trajectory(entry):
    episode=Path(entry["episode_path"]);processed=episode/"processed";manifest=_yaml(processed/"processing_manifest.yaml");config=_yaml(episode/"config_resolved.yaml")
    with np.load(processed/"synchronized_episode.npz",allow_pickle=False) as archive:data={name:archive[name] for name in archive.files}
    selected=np.flatnonzero(data["segment_id"]==int(entry["segment_id"]))
    if not len(selected):raise ValueError("empty valid segment")
    sources=manifest["source_files"];primary=decode_selected_rgb(sources["primary_video"],data["primary_frame_indices"][selected])
    primary=np.stack([center_crop_resize(x,config["dataset_preprocessing"]["primary"]["output_resolution"]) for x in primary]).astype(np.uint8)
    wrist_indices=data["wrist_frame_indices"][selected]
    if sources.get("wrist_video") and np.all(wrist_indices>=0):
        wrist=decode_selected_rgb(sources["wrist_video"],wrist_indices);wrist=np.stack([center_crop_resize(x,config["dataset_preprocessing"]["wrist"]["output_resolution"]) for x in wrist]).astype(np.uint8);wrist_valid=np.ones(len(selected),bool)
    else:
        h,w=config["dataset_preprocessing"]["wrist"]["output_resolution"];wrist=np.zeros((len(selected),h,w,3),np.uint8);wrist_valid=np.zeros(len(selected),bool)
    proprio=np.concatenate([data["joint_positions"][selected],data["tcp_positions"][selected],data["tcp_quaternions_xyzw"][selected],data["gripper_states"][selected,None]],axis=1).astype(np.float32);instruction=_json(episode/"manifest.json")["instruction"]
    steps=[]
    for local,index in enumerate(selected):steps.append({"observation":{"image_primary":primary[local],"image_wrist":wrist[local],"wrist_valid":bool(wrist_valid[local]),"proprio":proprio[local],"joint_position":data["joint_positions"][index],"tcp_pose":np.concatenate([data["tcp_positions"][index],data["tcp_quaternions_xyzw"][index]]).astype(np.float32),"gripper_state":np.int64(data["gripper_states"][index])},"action":data["actions"][index].astype(np.float32),"language_instruction":instruction,"is_first":bool(local==0),"is_last":bool(local==len(selected)-1),"is_terminal":bool(local==len(selected)-1)})
    return f"{entry['episode_id']}_segment_{entry['segment_id']:03d}",{"steps":steps,"episode_metadata":{"episode_id":entry["episode_id"],"segment_id":np.int64(entry["segment_id"]),"quality_grade":entry["quality_grade"]}}
