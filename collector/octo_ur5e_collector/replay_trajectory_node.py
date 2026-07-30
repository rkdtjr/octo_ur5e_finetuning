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
    if (root/"replay/rosbag2").exists():
        raise RuntimeError(
            "replay bag already exists; overwrite forbidden. Use a new episode or "
            "archive the interrupted replay/rosbag2 directory before retrying."
        )
    # Deliberately require the ROS runtime implementation and explicit fallback.
    from .ros_adapters.rosbag_recorder import RosbagRecorder
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as e: raise RuntimeError(f"ROS Python packages unavailable: {e}") from e
    from rclpy.signals import SignalHandlerOptions
    d=config.data; rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=Node("octo_replay_trajectory"); latest={"q":None}
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
    bag.start(); update_status(root,"replaying"); start=time.monotonic(); trajectory_start=None; failure=None; goal_accepted=False; result_code=None; handle=None
    try:
        first=grip.command_semantic(int(g[0]),execute=True)
        if not first.success: raise RuntimeError("initial gripper command failed")
        time.sleep(d["replay"]["start_settle_sec"])
        from .ros_adapters.trajectory_client import TrajectoryClient
        feedback={"progress":None,"receipt":None,"max_error":None}
        def feedback_cb(message):
            point=message.feedback
            duration=point.desired.time_from_start
            feedback["progress"]=duration.sec+duration.nanosec/1e9
            feedback["receipt"]=time.monotonic()
            if point.error.positions:
                feedback["max_error"]=float(np.max(np.abs(point.error.positions)))
        client=TrajectoryClient(node,d["ros"]["trajectory_action"],d["replay"]["controller_joint_order"],feedback_cb)
        goal=client.make_goal(times,q)
        future=client.client.send_goal_async(goal,feedback_callback=feedback_cb)
        rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        handle=future.result()
        if handle is None or not handle.accepted: raise RuntimeError("trajectory goal rejected")
        goal_accepted=True; trajectory_start=time.monotonic(); pending=list(events)
        fallback_timeout=float(times[-1])*d["replay"]["result_timeout_factor"]+d["replay"]["result_timeout_margin_sec"]
        result_deadline=None; end_reached=False; last_progress=-1
        print(
            "trajectory accepted; controller feedback will determine completion "
            f"(no-feedback fallback timeout={fallback_timeout:.1f}s)",
            flush=True,
        )
        result=handle.get_result_async()
        while not result.done():
            rclpy.spin_once(node,timeout_sec=.01); elapsed=time.monotonic()-trajectory_start
            controller_progress=feedback["progress"]
            trigger_progress=controller_progress if controller_progress is not None else elapsed
            second=int(elapsed)
            if second!=last_progress:
                last_progress=second
                progress_text="no feedback" if controller_progress is None else f"controller={controller_progress:.2f}s"
                error_text="" if feedback["max_error"] is None else f" max_error={feedback['max_error']:.4f}rad"
                scale_text=""
                if controller_progress is not None and elapsed>0:
                    effective_scale=controller_progress/elapsed
                    eta=(float(times[-1])-controller_progress)/effective_scale if effective_scale>1e-6 else float("inf")
                    scale_text=f" scale={effective_scale:.2f} ETA={max(0,eta):.1f}s"
                print(f"[wall={elapsed:6.1f}s / planned={times[-1]:.1f}s] {progress_text}{scale_text}{error_text}",flush=True)
            if feedback["receipt"] is not None and time.monotonic()-feedback["receipt"]>d["replay"]["feedback_stale_sec"]:
                raise RuntimeError(
                    f"trajectory feedback stale for more than {d['replay']['feedback_stale_sec']:.1f}s; "
                    f"last controller progress={controller_progress}"
                )
            while pending and pending[0]["time_sec"]/d["replay"]["speed_scale"]<=trigger_progress:
                ev=pending.pop(0); command=grip.command_semantic(ev["semantic_state"],execute=True)
                if not command.success: handle.cancel_goal_async(); raise RuntimeError("gripper replay command failed")
            if controller_progress is not None and controller_progress>=float(times[-1])-1e-3:
                if not end_reached:
                    end_reached=True
                    result_deadline=time.monotonic()+d["replay"]["result_timeout_margin_sec"]
            elif controller_progress is not None:
                result_deadline=None
            elif controller_progress is None and result_deadline is None:
                result_deadline=trajectory_start+fallback_timeout
            if result_deadline is not None and time.monotonic()>=result_deadline:
                raise RuntimeError(
                    "trajectory result timeout; "
                    f"last controller progress={controller_progress}, max_error={feedback['max_error']}"
                )
        result_code=result.result().result.error_code
        print(f"trajectory result received: error_code={result_code}",flush=True)
        time.sleep(d["replay"]["end_settle_sec"])
    except (KeyboardInterrupt,Exception) as e:
        failure=str(e)
        if isinstance(e,KeyboardInterrupt): failure="Ctrl+C"
        if handle is not None and rclpy.ok():
            cancel=handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node,cancel,timeout_sec=2)
            print(f"trajectory cancel requested: {failure}",flush=True)
    finally:
        bag_code=bag.stop(); actual=time.monotonic()-start; node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    summary={"execute":True,"goal_accepted":goal_accepted,"result_code":result_code,"planned_duration_sec":float(times[-1]),"actual_duration_sec":actual,"joint_tracking_rmse":None,"joint_tracking_max_error":None,"gripper_event_count":len(events),"gripper_event_timing_errors":[],"camera_topic_message_counts":{},"state_topic_message_counts":{},"bag_storage_id":d["storage"]["rosbag_storage_id"],"bag_exit_code":bag_code,"failure_reason":failure,"timing_mode":"wall_clock_explicit_fallback"}
    (root/"replay/execution_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    update_status(root,"failed" if failure or result_code!=0 else "completed",failure)
    return 2 if failure or result_code!=0 else 0

def main():
    from .collector_cli import main as cli
    cli(["replay",*sys.argv[1:]])

if __name__ == "__main__":
    main()
