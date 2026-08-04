from __future__ import annotations
import json,sys,time,shutil
from pathlib import Path
import numpy as np
from .core.episode import update_status,utc_now
from .core.trajectory import validate_arrays,validate_trajectory_file
from .ros_adapters.preflight import run_preflight,preflight_ok

def run_replay(root:Path,config,execute=False,wall_clock_fallback=False,move_to_start=False,move_to_start_duration_sec=8.0,return_to_start=False,return_to_start_duration_sec=8.0,smooth_trajectory=False,smoothing_window_sec=.35,smoothing_polyorder=3,gripper_anchor_window_sec=.5):
    command_wall_start=time.monotonic()
    validation=validate_trajectory_file(root,config,True)
    if not validation["valid"]:
        print(json.dumps(validation,indent=2)); return 2
    z=np.load(root/"demonstration/trajectory.npz"); times=z["time_from_start_sec"]/config.replay["speed_scale"]; q=z["joint_position"]; g=z["gripper_semantic_state"]
    events=list(validation["gripper_transitions"])
    open_state=int(config.gripper["semantic_open"])
    if int(g[0])!=open_state:
        events.insert(0,{"index":0,"time_sec":0.0,"semantic_state":int(g[0]),"reason":"restore_recorded_initial_state"})
    smoothing={"enabled":False}
    if smooth_trajectory:
        from .core.trajectory_smoothing import smooth_joint_trajectory
        anchor_times=[x["time_sec"]/config.replay["speed_scale"] for x in validation["gripper_transitions"]]
        q,smoothing=smooth_joint_trajectory(times,q,anchor_times,smoothing_window_sec,smoothing_polyorder,gripper_anchor_window_sec)
        smoothed_validation=validate_arrays(times,q,g,z["tcp_pose6"],config.replay["max_joint_velocity_rad_s"],config.replay["max_joint_acceleration_rad_s2"])
        if not smoothed_validation["valid"]:raise RuntimeError(f"smoothed trajectory invalid: {smoothed_validation['errors']}")
        print(f"SMOOTHING max_change={smoothing['max_position_change_rad']:.5f}rad velocity={smoothing['before']['max_velocity_rad_s']:.4f}->{smoothing['after']['max_velocity_rad_s']:.4f}rad/s acceleration={smoothing['before']['max_acceleration_rad_s2']:.2f}->{smoothing['after']['max_acceleration_rad_s2']:.2f}rad/s^2 anchors={smoothing['anchor_count']}",flush=True)
    print(f"{'EXECUTE' if execute else 'DRY-RUN'} points={len(times)} duration={times[-1]:.3f}s max_velocity={validation['max_velocity_rad_s']:.4f} gripper_events={events}")
    if not execute:
        print("No trajectory action or SetIO service was called."); return 0
    checks=run_preflight(config,True,True)
    for c in checks: print(f"{'OK' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    if not preflight_ok(checks): return 2
    if not wall_clock_fallback:
        raise RuntimeError("controller-feedback gripper scheduling is UNVERIFIED_ON_HARDWARE; pass --wall-clock-gripper-fallback explicitly")
    replay_root=root/"replay"
    existing=[replay_root/x for x in ("robot_states.mcap","primary.mkv","wrist.mkv","primary.mp4","wrist.mp4")]
    if any(x.exists() for x in existing) or (replay_root/"robot_states_bag").exists():
        raise RuntimeError(
            "replay capture already exists; overwrite forbidden. Use a new episode or "
            "archive existing robot_states/video files before retrying."
        )
    # Deliberately require the ROS runtime implementation and explicit fallback.
    from .ros_adapters.rosbag_recorder import RosbagRecorder
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from tf2_ros import Buffer,TransformListener
    except ImportError as e: raise RuntimeError(f"ROS Python packages unavailable: {e}") from e
    from rclpy.signals import SignalHandlerOptions
    d=config.data; rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=Node("octo_replay_trajectory"); latest={"q":None,"velocity":None,"source_ros_ns":None}
    from .ros_adapters.freedrive_controller import FreedriveController
    controller_manager=FreedriveController(node,d["freedrive"])
    try:
        controller_manager.ensure_motion_controller()
        print(f"trajectory controller active: {d['freedrive']['motion_controller_name']}",flush=True)
    except Exception:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        raise
    runtime={"grip":None,"gripper_adapter":None,"feedback":None,"state":"preflight","gripper_source_ros_ns":None}
    tracking_errors=[];tcp_ages_ms=[]
    def cb(msg):
        idx={n:i for i,n in enumerate(msg.name)}
        if all(n in idx for n in d["robot"]["joint_names"]):
            latest["q"]=np.array([msg.position[idx[n]] for n in d["robot"]["joint_names"]])
            latest["velocity"]=[msg.velocity[idx[n]] if idx[n]<len(msg.velocity) else None for n in d["robot"]["joint_names"]]
            latest["source_ros_ns"]=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
    node.create_subscription(JointState,d["ros"]["joint_state_topic"],cb,20)
    tf_buffer=Buffer();tf_listener=TransformListener(tf_buffer,node)
    state_pub=node.create_publisher(String,"/octo_collector/robot_state",20)
    def publish_robot_state():
        if latest["q"] is None:return
        now=node.get_clock().now().nanoseconds;tcp=None;tcp_source_ros_ns=None;tcp_age_ms=None;tcp_valid=False
        try:
            tf=tf_buffer.lookup_transform(d["robot"]["base_frame"],d["robot"]["tcp_frame"],rclpy.time.Time())
            t=tf.transform.translation;r=tf.transform.rotation
            tcp_source_ros_ns=tf.header.stamp.sec*1_000_000_000+tf.header.stamp.nanosec
            tcp_age_ms=max(0,(now-tcp_source_ros_ns)/1e6)
            tcp_valid=tcp_age_ms<=d["synchronization"]["max_tcp_age_ms"]
            tcp_ages_ms.append(tcp_age_ms)
            if tcp_valid:tcp={"position":[t.x,t.y,t.z],"quaternion_xyzw":[r.x,r.y,r.z,r.w]}
        except Exception:pass
        fb=runtime["feedback"] or {}
        commanded=fb.get("desired_positions")
        if runtime["state"]=="executing" and commanded is not None:
            tracking_errors.append((latest["q"]-np.asarray(commanded,dtype=float)).tolist())
        readback=runtime["gripper_adapter"].latest_readback() if runtime["gripper_adapter"] else None
        gripper_source_ros_ns=readback.receipt_ros_ns if readback is not None else runtime["gripper_source_ros_ns"]
        payload={"ros_stamp_ns":now,"joint_source_ros_ns":latest["source_ros_ns"],"tcp_source_ros_ns":tcp_source_ros_ns,"tcp_age_ms":tcp_age_ms,"tcp_valid":tcp_valid,"monotonic_ns":time.monotonic_ns(),"actual_joint_positions":latest["q"].tolist(),"actual_joint_velocities":latest["velocity"],"actual_tcp":tcp,"gripper_state":runtime["grip"].last_state if runtime["grip"] else None,"gripper_physical_output_value":readback.value if readback is not None else None,"gripper_source_ros_ns":gripper_source_ros_ns,"replay_state":runtime["state"],"commanded_joint_positions":commanded}
        msg=String();msg.data=json.dumps(payload,separators=(",",":"));state_pub.publish(msg)
    node.create_timer(1/d["sampling"]["demonstration_rate_hz"],publish_robot_state)
    deadline=time.monotonic()+2
    while latest["q"] is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
    if latest["q"] is None:
        node.destroy_node();rclpy.shutdown();raise RuntimeError("no current joint state received")
    initial_error=float(np.max(np.abs(latest["q"]-q[0])))
    if initial_error>d["replay"]["initial_joint_tolerance_rad"]:
        if not move_to_start:
            node.destroy_node();rclpy.shutdown()
            raise RuntimeError(
                f"robot is not within initial joint tolerance (max error={initial_error:.4f} rad); "
                "pass --move-to-start explicitly to command a joint-space positioning move"
            )
        delta=np.abs(latest["q"]-q[0]);estimated_velocity=float(np.max(delta)/move_to_start_duration_sec)
        print(
            "WARNING: moving to recorded start in joint space; no collision checking is performed.\n"
            f"duration={move_to_start_duration_sec:.1f}s max_joint_delta={float(np.max(delta)):.4f}rad "
            f"estimated_max_velocity={estimated_velocity:.4f}rad/s",
            flush=True,
        )
        from .ros_adapters.trajectory_client import TrajectoryClient
        positioning_feedback={"progress":0.0}
        def positioning_cb(message):
            duration=message.feedback.desired.time_from_start
            positioning_feedback["progress"]=duration.sec+duration.nanosec/1e9
        positioning=TrajectoryClient(node,d["ros"]["trajectory_action"],d["replay"]["controller_joint_order"],positioning_cb)
        goal=positioning.make_goal([move_to_start_duration_sec],[q[0]])
        send=positioning.client.send_goal_async(goal,feedback_callback=positioning_cb)
        rclpy.spin_until_future_complete(node,send,timeout_sec=5)
        handle=send.result()
        if handle is None or not handle.accepted:
            node.destroy_node();rclpy.shutdown();raise RuntimeError("move-to-start goal rejected")
        print("move-to-start goal accepted; Ctrl+C to cancel",flush=True)
        result=handle.get_result_async();last_second=-1
        try:
            deadline=time.monotonic()+move_to_start_duration_sec*10+10
            while not result.done():
                rclpy.spin_once(node,timeout_sec=.02)
                second=int(positioning_feedback["progress"])
                if second!=last_second:
                    last_second=second;print(f"positioning progress={positioning_feedback['progress']:.2f}/{move_to_start_duration_sec:.2f}s",flush=True)
                if time.monotonic()>deadline:raise RuntimeError("move-to-start result timeout")
        except (KeyboardInterrupt,Exception) as e:
            cancel=handle.cancel_goal_async();rclpy.spin_until_future_complete(node,cancel,timeout_sec=2)
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
            if isinstance(e,KeyboardInterrupt):return 130
            raise
        if result.result().result.error_code!=0:
            code=result.result().result.error_code;node.destroy_node();rclpy.shutdown();raise RuntimeError(f"move-to-start failed with error_code={code}")
        verify_deadline=time.monotonic()+1.0
        while time.monotonic()<verify_deadline:rclpy.spin_once(node,timeout_sec=.05)
        initial_error=float(np.max(np.abs(latest["q"]-q[0])))
        if initial_error>d["replay"]["initial_joint_tolerance_rad"]:
            node.destroy_node();rclpy.shutdown();raise RuntimeError(f"move-to-start completed but tolerance check failed: {initial_error:.4f} rad")
        print(f"move-to-start complete; max joint error={initial_error:.5f}rad",flush=True)
    from .ros_adapters.digital_output_gripper import DigitalOutputGripperAdapter
    from .core.gripper import GripperController
    adapter=DigitalOutputGripperAdapter(node,{**d["gripper"],**d["ros"]}); grip=GripperController(d["gripper"],lambda value:adapter.send(value,d["gripper"]["command_timeout_sec"]))
    runtime["grip"]=grip;runtime["gripper_adapter"]=adapter
    bag=RosbagRecorder(
        replay_root/"robot_states_bag",d["raw_topics"]["replay"],
        d["storage"]["rosbag_storage_id"],d["storage"]["rosbag_storage_preset_profile"],
    )
    from .ros_adapters.camera_video_recorder import CameraVideoRecorder
    video_recorders=[]
    cr=d["camera_recording"];container=d["storage"]["video_container"]
    for name in ("primary","wrist"):
        if cr[name]["enabled"]:
            video_recorders.append(CameraVideoRecorder(node,name,cr[name],replay_root,cr["capture_fps"],cr["encoder_queue_size"],container))
    for recorder in video_recorders:recorder.start()
    bag.start();tracking_errors.clear();tcp_ages_ms.clear(); update_status(root,"replaying"); recording_start=time.monotonic(); trajectory_start=None;trajectory_execution_duration=None; failure=None; goal_accepted=False; result_code=None; handle=None;return_to_start_completed=False
    replay_events=[]
    events_file=open(replay_root/"events.jsonl","a",encoding="utf-8")
    def spin_for(seconds):
        deadline=time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=min(.02,deadline-time.monotonic()))
    try:
        first=grip.command_semantic(open_state,execute=True)
        if not first.success: raise RuntimeError("initial gripper command failed")
        runtime["gripper_source_ros_ns"]=node.get_clock().now().nanoseconds
        print(
            f"gripper initialized OPEN (semantic={open_state}, "
            f"DO{d['gripper']['output_pin']}={first.output_value:g}, readback confirmed)",
            flush=True,
        )
        print(f"waiting {d['gripper']['actuation_settle_sec']:.1f}s for gripper actuation...",flush=True)
        spin_for(d["gripper"]["actuation_settle_sec"])
        from .ros_adapters.trajectory_client import TrajectoryClient
        feedback={"progress":None,"receipt":None,"max_error":None,"desired_positions":None}
        runtime["feedback"]=feedback
        def feedback_cb(message):
            point=message.feedback
            duration=point.desired.time_from_start
            feedback["progress"]=duration.sec+duration.nanosec/1e9
            feedback["receipt"]=time.monotonic()
            if point.error.positions:
                feedback["max_error"]=float(np.max(np.abs(point.error.positions)))
            if point.desired.positions:feedback["desired_positions"]=list(point.desired.positions)
        client=TrajectoryClient(node,d["ros"]["trajectory_action"],d["replay"]["controller_joint_order"],feedback_cb)
        goal=client.make_goal(times,q)
        future=client.client.send_goal_async(goal,feedback_callback=feedback_cb)
        rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        handle=future.result()
        if handle is None or not handle.accepted: raise RuntimeError("trajectory goal rejected")
        goal_accepted=True;runtime["state"]="executing";trajectory_start=time.monotonic();pending=list(events)
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
                ev=pending.pop(0);scheduled=float(ev["time_sec"])/d["replay"]["speed_scale"]
                if int(ev["semantic_state"])==grip.last_state:
                    continue
                actual_command_time=float(trigger_progress); command=grip.command_semantic(ev["semantic_state"],execute=True)
                status=adapter.last_command_status
                runtime["gripper_source_ros_ns"]=node.get_clock().now().nanoseconds
                from .core.quality_metrics import gripper_event_record
                replay_event=gripper_event_record(scheduled,actual_command_time,ev["semantic_state"],command.output_value,d["gripper"]["output_pin"],status.service_success,status.readback_confirmed)
                replay_events.append(replay_event);events_file.write(json.dumps(replay_event)+"\n");events_file.flush()
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
        trajectory_execution_duration=time.monotonic()-trajectory_start
        runtime["state"]="settling"
        print(f"trajectory result received: error_code={result_code}",flush=True)
        spin_for(d["replay"]["end_settle_sec"])
        final_open=grip.command_semantic(open_state,execute=True)
        if not final_open.success:raise RuntimeError("failed to open gripper after replay")
        print(
            f"final gripper OPEN (semantic={open_state}, "
            f"DO{d['gripper']['output_pin']}={final_open.output_value:g})",
            flush=True,
        )
        spin_for(d["gripper"]["actuation_settle_sec"])
        if return_to_start:
            runtime["state"]="returning_to_start"
            print(
                "WARNING: returning to recorded start in joint space; no collision checking is performed.\n"
                f"duration={return_to_start_duration_sec:.1f}s",
                flush=True,
            )
            return_goal=client.make_goal([return_to_start_duration_sec],[q[0]])
            return_future=client.client.send_goal_async(return_goal)
            rclpy.spin_until_future_complete(node,return_future,timeout_sec=5)
            return_handle=return_future.result()
            if return_handle is None or not return_handle.accepted:
                raise RuntimeError("return-to-start goal rejected")
            handle=return_handle
            return_result=return_handle.get_result_async()
            return_deadline=time.monotonic()+return_to_start_duration_sec*10+10
            while not return_result.done():
                rclpy.spin_once(node,timeout_sec=.02)
                if time.monotonic()>return_deadline:
                    return_handle.cancel_goal_async()
                    raise RuntimeError("return-to-start result timeout")
            if return_result.result().result.error_code!=0:
                raise RuntimeError(f"return-to-start failed with error_code={return_result.result().result.error_code}")
            return_to_start_completed=True
            print("return-to-start complete",flush=True)
    except (KeyboardInterrupt,Exception) as e:
        failure=str(e)
        if isinstance(e,KeyboardInterrupt): failure="Ctrl+C"
        if handle is not None and rclpy.ok():
            cancel=handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node,cancel,timeout_sec=2)
            print(f"trajectory cancel requested: {failure}",flush=True)
    finally:
        if trajectory_start is not None and trajectory_execution_duration is None:
            trajectory_execution_duration=time.monotonic()-trajectory_start
        recording_stop=time.monotonic()
        events_file.close()
        camera_stats={}
        for recorder in video_recorders:
            try:camera_stats[recorder.name]=recorder.stop()
            except Exception as e:camera_stats[recorder.name]={"error":str(e)}
        bag_code=bag.stop(); recording_duration=recording_stop-recording_start
        try:bag.export_single_mcap(replay_root/"robot_states.mcap")
        except Exception as e:
            if failure is None:failure=f"robot state MCAP finalize failed: {e}"
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    metadata={"schema_version":4,"episode_id":root.name,"container":container,"capture_fps":cr["capture_fps"],"dataset_rate_hz":cr["dataset_rate_hz"],"robot_state_file":"robot_states.mcap","camera_timestamp_clock":"ROS header stamp with local monotonic receipt index","preview_recorded":False,"gripper":{"output_pin":d["gripper"]["output_pin"],"semantic_open":d["gripper"]["semantic_open"],"semantic_closed":d["gripper"]["semantic_closed"],"physical_output_for_open":d["gripper"]["output_value_for_open"],"physical_output_for_closed":d["gripper"]["output_value_for_closed"]},"training_data_contract":{"timeline":"common ROS nanoseconds","frame_selection":"nearest unique frame to 10 Hz target timestamp","camera_output":"RGB uint8","gripper_semantic":{"open":0,"closed":1},"tcp_orientation":"quaternion_xyzw","joint_order":d["robot"]["joint_names"]},"cameras":{}}
    metadata["trajectory_smoothing"]=smoothing
    for recorder in video_recorders:
        metadata["cameras"][recorder.name]={"file":f"{recorder.name}.{container}","timestamps":f"{recorder.name}_timestamps.csv","resolution":recorder.c["resolution"],"configured_source_encoding":recorder.c["source_encoding"],"observed_source_encodings":camera_stats[recorder.name].get("observed_source_encodings"),"source_topic":recorder.c["source_topic"],"bayer_pattern":recorder.c["bayer_pattern"],"encoder_input_color_order":"BGR","stored_pixel_format":recorder.c["pixel_format"],"decoded_dataset_contract":"RGB uint8","dataset_preprocessing":d["dataset_preprocessing"][recorder.name],"codec":"h264","encoder":camera_stats[recorder.name].get("encoder"),"bitrate_mbps":recorder.c["bitrate_mbps"],"maxrate_mbps":recorder.c["maxrate_mbps"],"bufsize_mbps":recorder.c["bufsize_mbps"],"gop_size":recorder.c["gop_size"]}
    (replay_root/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    from .core.video_recording import storage_projection
    files=[replay_root/f"{x}.{container}" for x in ("primary","wrist")]+[replay_root/"robot_states.mcap"]
    projection=storage_projection(files,recording_duration)
    quality={"cameras":camera_stats,**projection,"warnings":[]}
    for name,s in camera_stats.items():
        for source,target in (("actual_fps","actual_fps"),("frame_count","frame_count"),("frame_drop_ratio","frame_drop_ratio"),("interval_mean_ms","interval_mean_ms"),("interval_std_ms","interval_std_ms"),("max_interval_ms","max_interval_ms"),("bitrate_actual_mbps","bitrate_actual_mbps"),("file_size_bytes","file_size_bytes")):
            quality[f"{name}_{target}"]=s.get(source)
        if abs(s.get("actual_fps",0)-cr["capture_fps"])/cr["capture_fps"]>.1:quality["warnings"].append(f"{name}: actual FPS outside 10%")
        if s.get("frame_drop_ratio",0)>.01:quality["warnings"].append(f"{name}: frame drop ratio exceeds 1%")
        if s.get("max_interval_ms") and s["max_interval_ms"]>1.5*1000/cr["capture_fps"]:quality["warnings"].append(f"{name}: max frame interval exceeds 1.5x expected")
        if s.get("file_size_bytes",0)==0:quality["warnings"].append(f"{name}: video file is empty")
        if s.get("decoded_frame_count")!=s.get("timestamp_row_count"):quality["warnings"].append(f"{name}: decoded frame count differs from timestamp CSV")
        if s.get("error") or s.get("file_size_bytes",0)==0 or s.get("decoded_frame_count")!=s.get("timestamp_row_count"):
            if failure is None:failure=f"{name} video is not training-compatible"
    free=shutil.disk_usage(replay_root).free/1024**3
    quality["free_space_gib"]=free
    if free<d["storage"]["minimum_free_space_gib"]:quality["warnings"].append("disk free space below configured minimum")
    if projection["expected_size_for_100_episodes_gib"]>free:quality["warnings"].append("estimated 100-episode size exceeds free space")
    from .core.quality_metrics import age_metrics,duration_metrics,joint_tracking_metrics
    tracking=joint_tracking_metrics(tracking_errors,d["robot"]["joint_names"])
    quality.update(age_metrics(tcp_ages_ms,"tcp_age"))
    try:
        from .core.synchronize_episode import synchronize_episode
        sync=synchronize_episode(replay_root,cr["dataset_rate_hz"],d["synchronization"]["max_camera_time_error_ms"],d["synchronization"])
        quality.update({k:v for k,v in sync.items() if k.endswith("_ms")})
    except Exception as e:
        sync={};quality["warnings"].append(f"synchronization metrics failed: {e}")
        if failure is None:failure=f"synchronization metrics failed: {e}"
    quality.update(tracking)
    total_command_wall_time=time.monotonic()-command_wall_start
    durations=duration_metrics(times[-1],trajectory_execution_duration,recording_duration,total_command_wall_time)
    summary={"schema_version":4,"execute":True,"goal_accepted":goal_accepted,"result_code":result_code,"dataset_compatible":failure is None and result_code==0,"return_to_start_requested":return_to_start,"return_to_start_completed":return_to_start_completed,**durations,**tracking,"gripper_event_count":len(replay_events),"gripper_event_timing_errors":[x["timing_error_sec"] for x in replay_events],"gripper_events":replay_events,"camera_topic_message_counts":{k:v.get("frame_count") for k,v in camera_stats.items()},"state_topic_message_counts":{"/octo_collector/robot_state":len(tcp_ages_ms)},"bag_storage_id":d["storage"]["rosbag_storage_id"],"bag_exit_code":bag_code,"failure_reason":failure,"timing_mode":"controller_feedback_with_explicit_wall_fallback"}
    summary["trajectory_smoothing"]=smoothing
    from .core.quality_metrics import evaluate_quality
    evaluation=evaluate_quality(quality,summary,d["synchronization"]["max_tcp_age_ms"])
    quality["evaluation"]=evaluation
    quality["quality_grade"]=evaluation["overall"]
    quality["quality_summary"]=f"{evaluation['overall']}: {evaluation['counts']['GOOD']} good, {evaluation['counts']['WARNING']} warning, {evaluation['counts']['BAD']} bad"
    summary["quality_grade"]=evaluation["overall"]
    summary["quality_verdict"]=evaluation["verdict"]
    summary["quality_problems"]=evaluation["problems"]
    (replay_root/"quality_report.json").write_text(json.dumps(quality,indent=2)+"\n")
    (root/"replay/execution_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (root/"replay/episode_result.json").write_text(json.dumps(summary,indent=2)+"\n")
    update_status(root,"failed" if failure or result_code!=0 else "completed",failure)
    return 2 if failure or result_code!=0 else 0

def main():
    from .collector_cli import main as cli
    cli(["replay",*sys.argv[1:]])

if __name__ == "__main__":
    main()
