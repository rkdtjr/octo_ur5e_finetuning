from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
from .core.episode import update_status,utc_now
from .core.trajectory import validate_trajectory_file
from .ros_adapters.preflight import run_preflight,preflight_ok

def run_replay(root:Path,config,execute=False,wall_clock_fallback=False):
    validation=validate_trajectory_file(root,config,True)
    if not validation["valid"]:
        print(json.dumps(validation,indent=2)); return 2
    z=np.load(root/"demonstration/trajectory.npz"); times=z["time_from_start_sec"]/config.replay["speed_scale"]; q=z["joint_position"]; g=z["gripper_semantic_state"]
    events=validation["gripper_transitions"]
    print(f"{'EXECUTE' if execute else 'DRY-RUN'} points={len(times)} duration={times[-1]:.3f}s max_velocity={validation['max_velocity_rad_s']:.4f} gripper_events={events}")
    if not execute:
        print("No trajectory action or SetIO service was called."); return 0
    checks=run_preflight(config,True,True)
    for c in checks: print(f"{'OK' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    if not preflight_ok(checks): return 2
    if not wall_clock_fallback:
        raise RuntimeError("controller-feedback gripper scheduling is UNVERIFIED_ON_HARDWARE; pass --wall-clock-gripper-fallback explicitly")
    # Deliberately require the ROS runtime implementation and explicit fallback.
    from .ros_adapters.rosbag_recorder import RosbagRecorder
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as e: raise RuntimeError(f"ROS Python packages unavailable: {e}") from e
    d=config.data; rclpy.init(); node=Node("octo_replay_trajectory"); latest={"q":None}
    def cb(msg):
        idx={n:i for i,n in enumerate(msg.name)}
        if all(n in idx for n in d["robot"]["joint_names"]):latest["q"]=np.array([msg.position[idx[n]] for n in d["robot"]["joint_names"]])
    node.create_subscription(JointState,d["ros"]["joint_state_topic"],cb,20)
    deadline=time.monotonic()+2
    while latest["q"] is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
    if latest["q"] is None or np.max(np.abs(latest["q"]-q[0]))>d["replay"]["initial_joint_tolerance_rad"]:
        node.destroy_node(); rclpy.shutdown(); raise RuntimeError("robot is not within initial joint tolerance; automatic positioning is forbidden")
    from .ros_adapters.digital_output_gripper import DigitalOutputGripperAdapter
    from .core.gripper import GripperController
    adapter=DigitalOutputGripperAdapter(node,{**d["gripper"],**d["ros"]}); grip=GripperController(d["gripper"],lambda value:adapter.send(value,d["gripper"]["command_timeout_sec"]))
    bag=RosbagRecorder(root/"replay/rosbag2",d["raw_topics"]["replay"],d["storage"]["rosbag_storage_id"])
    if (root/"replay/rosbag2").exists(): raise RuntimeError("replay bag already exists; overwrite forbidden")
    bag.start(); update_status(root,"replaying"); start=time.monotonic(); trajectory_start=None; failure=None; goal_accepted=False; result_code=None; handle=None
    try:
        first=grip.command_semantic(int(g[0]),execute=True)
        if not first.success: raise RuntimeError("initial gripper command failed")
        time.sleep(d["replay"]["start_settle_sec"])
        from .ros_adapters.trajectory_client import TrajectoryClient
        client=TrajectoryClient(node,d["ros"]["trajectory_action"],d["replay"]["controller_joint_order"])
        goal=client.make_goal(times,q); future=client.client.send_goal_async(goal); rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        handle=future.result()
        if handle is None or not handle.accepted: raise RuntimeError("trajectory goal rejected")
        goal_accepted=True; trajectory_start=time.monotonic(); pending=list(events)
        result=handle.get_result_async()
        while not result.done():
            rclpy.spin_once(node,timeout_sec=.01); elapsed=time.monotonic()-trajectory_start
            while pending and pending[0]["time_sec"]/d["replay"]["speed_scale"]<=elapsed:
                ev=pending.pop(0); command=grip.command_semantic(ev["semantic_state"],execute=True)
                if not command.success: handle.cancel_goal_async(); raise RuntimeError("gripper replay command failed")
        result_code=result.result().result.error_code
        time.sleep(d["replay"]["end_settle_sec"])
    except (KeyboardInterrupt,Exception) as e:
        failure=str(e)
        if isinstance(e,KeyboardInterrupt): failure="Ctrl+C"
        if handle is not None: handle.cancel_goal_async()
    finally:
        bag_code=bag.stop(); actual=time.monotonic()-start; node.destroy_node(); rclpy.shutdown()
    summary={"execute":True,"goal_accepted":goal_accepted,"result_code":result_code,"planned_duration_sec":float(times[-1]),"actual_duration_sec":actual,"joint_tracking_rmse":None,"joint_tracking_max_error":None,"gripper_event_count":len(events),"gripper_event_timing_errors":[],"camera_topic_message_counts":{},"state_topic_message_counts":{},"bag_storage_id":d["storage"]["rosbag_storage_id"],"bag_exit_code":bag_code,"failure_reason":failure,"timing_mode":"wall_clock_explicit_fallback"}
    (root/"replay/execution_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    update_status(root,"failed" if failure or result_code!=0 else "completed",failure)
    return 2 if failure or result_code!=0 else 0

def main():
    from .collector_cli import main as cli
    cli(["replay",*sys.argv[1:]])

if __name__ == "__main__":
    main()
