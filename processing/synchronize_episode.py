"""Build a timestamp-based 10 Hz camera synchronization index."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from octo_ur5e_collector.core.synchronization import match_cameras
from octo_ur5e_collector.core.synchronization import interpolate_linear,interpolate_quaternion,latest_discrete

def read_stamps(path):
    with open(path,newline="") as f:return np.array([int(row["ros_stamp_ns"]) for row in csv.DictReader(f)],dtype=np.int64)

def read_robot_states(path):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import String
        reader=rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(path),storage_id="mcap"),rosbag2_py.ConverterOptions("",""))
        states=[]
        while reader.has_next():
            topic,data,_=reader.read_next()
            if topic=="/octo_collector/robot_state":states.append(json.loads(deserialize_message(data,String).data))
        return states
    except Exception:return []

def robot_match(states,target_ns,thresholds):
    usable=[x for x in states if x.get("actual_tcp") and x.get("actual_joint_positions") and x.get("gripper_state") in (0,1)]
    if len(usable)<2:return None
    times=np.array([x["ros_stamp_ns"] for x in usable],dtype=np.int64)
    try:
        pos,pose_gap=interpolate_linear(times,[x["actual_tcp"]["position"] for x in usable],target_ns)
        quat,_=interpolate_quaternion(times,[x["actual_tcp"]["quaternion_xyzw"] for x in usable],target_ns)
        joints,joint_gap=interpolate_linear(times,[x["actual_joint_positions"] for x in usable],target_ns)
        grip,grip_age=latest_discrete(times,[x["gripper_state"] for x in usable],target_ns)
    except ValueError:return None
    valid=pose_gap<=thresholds["max_pose_interpolation_gap_ms"]*1e6 and joint_gap<=thresholds["max_joint_interpolation_gap_ms"]*1e6 and grip_age<=thresholds["max_gripper_age_ms"]*1e6
    return {"tcp_position":pos.tolist(),"tcp_quaternion_xyzw":quat.tolist(),"joint_position":joints.tolist(),"gripper_state":int(grip),"pose_interpolation_gap_ms":pose_gap/1e6,"joint_interpolation_gap_ms":joint_gap/1e6,"gripper_age_ms":grip_age/1e6,"robot_valid":bool(valid)}

def synchronize_episode(replay_dir,rate_hz=10.0,max_error_ms=40.0,thresholds=None):
    root=Path(replay_dir);p=read_stamps(root/"primary_timestamps.csv");w=read_stamps(root/"wrist_timestamps.csv")
    thresholds=thresholds or {"max_pose_interpolation_gap_ms":30.0,"max_joint_interpolation_gap_ms":30.0,"max_gripper_age_ms":100.0}
    states=read_robot_states(root/"robot_states.mcap")
    result=match_cameras(p,w,rate_hz,int(max_error_ms*1e6))
    out=root/"synchronization_index.csv"
    pose_gaps=[];joint_gaps=[];gripper_ages=[];final_valid_count=0
    with out.open("w",newline="") as f:
        names=["target_ns","primary_index","wrist_index","primary_time_error_ms","wrist_time_error_ms","primary_wrist_time_difference_ms","tcp_position","tcp_quaternion_xyzw","joint_position","gripper_state","pose_interpolation_gap_ms","joint_interpolation_gap_ms","gripper_age_ms","valid"]
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for i,target in enumerate(result["target_ns"]):
            pi,wi=int(result["primary_index"][i]),int(result["wrist_index"][i])
            robot=robot_match(states,int(target),thresholds)
            if robot:
                pose_gaps.append(robot["pose_interpolation_gap_ms"]);joint_gaps.append(robot["joint_interpolation_gap_ms"]);gripper_ages.append(robot["gripper_age_ms"])
            final_valid=bool(result["valid"][i] and robot is not None and robot["robot_valid"]);final_valid_count+=int(final_valid)
            writer.writerow({"target_ns":int(target),"primary_index":pi,"wrist_index":wi,"primary_time_error_ms":None if pi<0 else (int(p[pi])-int(target))/1e6,"wrist_time_error_ms":None if wi<0 else (int(w[wi])-int(target))/1e6,"primary_wrist_time_difference_ms":None if pi<0 or wi<0 else abs(int(p[pi])-int(w[wi]))/1e6,"tcp_position":None if robot is None else json.dumps(robot["tcp_position"]),"tcp_quaternion_xyzw":None if robot is None else json.dumps(robot["tcp_quaternion_xyzw"]),"joint_position":None if robot is None else json.dumps(robot["joint_position"]),"gripper_state":None if robot is None else robot["gripper_state"],"pose_interpolation_gap_ms":None if robot is None else robot["pose_interpolation_gap_ms"],"joint_interpolation_gap_ms":None if robot is None else robot["joint_interpolation_gap_ms"],"gripper_age_ms":None if robot is None else robot["gripper_age_ms"],"valid":final_valid})
    diffs=np.array([abs(int(p[int(pi)])-int(w[int(wi)]))/1e6 for pi,wi in zip(result["primary_index"],result["wrist_index"]) if pi>=0 and wi>=0])
    summary={"schema_version":3,"sample_count":len(result["target_ns"]),"valid_count":final_valid_count,"rate_hz":rate_hz,"selection":"nearest timestamp without duplicate source frames","index_file":out.name,"primary_wrist_time_difference_mean_ms":float(diffs.mean()) if len(diffs) else None,"primary_wrist_time_difference_p95_ms":float(np.percentile(diffs,95)) if len(diffs) else None,"primary_wrist_time_difference_max_ms":float(diffs.max()) if len(diffs) else None,"pose_sync_p95_ms":float(np.percentile(pose_gaps,95)) if pose_gaps else None,"joint_sync_p95_ms":float(np.percentile(joint_gaps,95)) if joint_gaps else None,"gripper_age_p95_ms":float(np.percentile(gripper_ages,95)) if gripper_ages else None}
    (root/"synchronization_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

def main():
    p=argparse.ArgumentParser();p.add_argument("replay_dir");p.add_argument("--rate-hz",type=float,default=10);p.add_argument("--max-error-ms",type=float,default=40)
    a=p.parse_args();print(json.dumps(synchronize_episode(a.replay_dir,a.rate_hz,a.max_error_ms),indent=2))
if __name__=="__main__":main()
