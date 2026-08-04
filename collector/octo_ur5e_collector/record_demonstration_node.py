from __future__ import annotations
import json,select,sys,termios,threading,time,tty
import numpy as np
from .core.episode import create_episode,update_status,utc_now
from .core.gripper import GripperController
from .core.keyboard_commands import KeyboardCommandQueue
from .core.trajectory import validate_arrays
from .core.transforms import matrix_to_quaternion_pose,matrix_to_ur_pose,quaternion_pose_to_matrix,relative_pose_action,ur_pose_to_matrix
from .ros_adapters.preflight import run_preflight,preflight_ok
from .ros_adapters.rosbag_recorder import RosbagRecorder

def _keyboard(queue,stop):
    if not sys.stdin.isatty(): return
    fd=sys.stdin.fileno(); old=termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            ready,_,_=select.select([sys.stdin],[],[],0.1)
            if ready:
                ch=sys.stdin.read(1)
                if ch=="\x1b": ch="esc"
                queue.submit(ch)
    finally: termios.tcsetattr(fd,termios.TCSADRAIN,old)

def run_recording(config,instruction,execute=False,initial_gripper=None,enable_freedrive=False,return_to_start=False,return_to_start_duration_sec=8.0,move_to_fixed_start=False):
    if initial_gripper=="closed":
        raise ValueError("recording must start with the gripper open")
    checks=run_preflight(config,execute,False,enable_freedrive,return_to_start or move_to_fixed_start)
    for c in checks: print(f"{'OK' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    if not preflight_ok(checks): return 2
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer,TransformListener
    except ImportError as e: raise RuntimeError(f"ROS Python packages unavailable: {e}") from e
    root=create_episode(config,instruction); update_status(root,"recording_demo")
    d=config.data; bag=RosbagRecorder(
        root/"demonstration/rosbag2",d["raw_topics"]["demonstration"],
        d["storage"]["rosbag_storage_id"],d["storage"]["rosbag_storage_preset_profile"],
    )
    rclpy.init(); node=Node("octo_record_demonstration")
    def spin_for(seconds):
        deadline=time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=min(.05,deadline-time.monotonic()))
    joint={"msg":None,"receipt":0}; samples=[]; q=KeyboardCommandQueue(d["keyboard"]); stop=threading.Event(); aborted=False; failure=None;return_to_start_completed=False
    def joint_cb(msg): joint.update(msg=msg,receipt=node.get_clock().now().nanoseconds)
    sub=node.create_subscription(JointState,d["ros"]["joint_state_topic"],joint_cb,50)
    tfbuf=Buffer(); listener=TransformListener(tfbuf,node)
    adapter=None
    if execute:
        from .ros_adapters.digital_output_gripper import DigitalOutputGripperAdapter
        adapter=DigitalOutputGripperAdapter(node,{**d["gripper"],**d["ros"]})
    freedrive=None
    grip=GripperController(d["gripper"],(lambda value:adapter.send(value,d["gripper"]["command_timeout_sec"])) if adapter else None)
    semantic=int(d["gripper"]["semantic_open"])
    initial_open=grip.command_semantic(semantic,execute=execute)
    if not initial_open.success:
        if freedrive is not None:
            try:freedrive.disable_and_restore()
            except Exception as restore_error:print(f"ERROR: failed to restore controller: {restore_error}",flush=True)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        update_status(root,"failed",f"could not initialize gripper open: {initial_open.reason}")
        raise RuntimeError(f"could not initialize gripper open: {initial_open.reason}")
    print(
        f"gripper initialized OPEN (semantic={semantic}, "
        f"DO{d['gripper']['output_pin']}={initial_open.output_value:g}, "
        f"{'readback confirmed' if execute else 'dry-run'})",
        flush=True,
    )
    print(f"waiting {d['gripper']['actuation_settle_sec']:.1f}s for gripper actuation...",flush=True)
    spin_for(d["gripper"]["actuation_settle_sec"])
    if move_to_fixed_start:
        try:
            from geometry_msgs.msg import PoseStamped
            from moveit_msgs.srv import GetPositionIK
            from .ros_adapters.freedrive_controller import FreedriveController
            from .ros_adapters.trajectory_client import TrajectoryClient
            controller=FreedriveController(node,d["freedrive"]);controller.ensure_motion_controller()
            deadline=time.monotonic()+5
            while joint["msg"] is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
            if joint["msg"] is None:raise RuntimeError("fixed-start move timed out waiting for joint state")
            target=ur_pose_to_matrix(d["recording_start"]["tcp_pose6"]);position,quat=matrix_to_quaternion_pose(target)
            ik=node.create_client(GetPositionIK,"/compute_ik")
            if not ik.wait_for_service(timeout_sec=3):raise RuntimeError("MoveIt /compute_ik unavailable; start move_group before recording")
            msg=joint["msg"];indices={name:i for i,name in enumerate(msg.name)}
            if any(name not in indices for name in d["robot"]["joint_names"]):raise RuntimeError("joint state missing configured UR joints")
            request=GetPositionIK.Request();request.ik_request.group_name=d["recording_start"]["move_group"];request.ik_request.avoid_collisions=True;request.ik_request.timeout.sec=2
            request.ik_request.robot_state.joint_state.name=list(d["robot"]["joint_names"]);request.ik_request.robot_state.joint_state.position=[msg.position[indices[name]] for name in d["robot"]["joint_names"]]
            request.ik_request.pose_stamped=PoseStamped();request.ik_request.pose_stamped.header.frame_id=d["robot"]["base_frame"];request.ik_request.pose_stamped.header.stamp=node.get_clock().now().to_msg();pose=request.ik_request.pose_stamped.pose
            pose.position.x,pose.position.y,pose.position.z=map(float,position);pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,quat)
            future=ik.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=3)
            if not future.done() or future.result() is None or future.result().error_code.val!=1:raise RuntimeError("collision-aware IK failed for configured recording start pose")
            solution=future.result().solution.joint_state;solution_index={name:i for i,name in enumerate(solution.name)};goal=np.asarray([solution.position[solution_index[name]] for name in d["replay"]["controller_joint_order"]],float)
            duration=float(d["recording_start"]["move_duration_sec"]);client=TrajectoryClient(node,d["ros"]["trajectory_action"],d["replay"]["controller_joint_order"],lambda _:None)
            if not client.client.wait_for_server(timeout_sec=3):raise RuntimeError("trajectory action unavailable for fixed-start move")
            print("WARNING: moving to configured recording start with collision-aware IK.",flush=True);print(f"target_tcp_pose6={d['recording_start']['tcp_pose6']} duration={duration:.1f}s",flush=True)
            sent=client.client.send_goal_async(client.make_goal([duration],[goal]));rclpy.spin_until_future_complete(node,sent,timeout_sec=5);handle=sent.result()
            if handle is None or not handle.accepted:raise RuntimeError("fixed-start trajectory goal rejected")
            result=handle.get_result_async();deadline=time.monotonic()+duration*3+5
            while not result.done():
                rclpy.spin_once(node,timeout_sec=.02)
                if time.monotonic()>deadline:handle.cancel_goal_async();raise RuntimeError("fixed-start trajectory result timeout")
            if result.result().result.error_code!=0:raise RuntimeError(f"fixed-start trajectory failed with error_code={result.result().result.error_code}")
            spin_for(.5);actual_tf=tfbuf.lookup_transform(d["robot"]["base_frame"],d["robot"]["tcp_frame"],rclpy.time.Time());tr=actual_tf.transform.translation;ro=actual_tf.transform.rotation
            actual=quaternion_pose_to_matrix([tr.x,tr.y,tr.z],[ro.x,ro.y,ro.z,ro.w]);error=relative_pose_action(target,actual,"base");position_error=float(np.linalg.norm(error[:3]));rotation_error=float(np.linalg.norm(error[3:]))
            if position_error>d["recording_start"]["position_tolerance_m"] or rotation_error>d["recording_start"]["rotation_tolerance_rad"]:raise RuntimeError(f"fixed-start tolerance failed: position={position_error:.4f}m rotation={rotation_error:.4f}rad")
            print(f"fixed recording start reached: position_error={position_error:.4f}m rotation_error={rotation_error:.4f}rad",flush=True)
        except Exception as error:
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
            update_status(root,"failed",str(error));raise
    if enable_freedrive:
        from .ros_adapters.freedrive_controller import FreedriveController
        freedrive=FreedriveController(node,d["freedrive"])
        try:
            freedrive.enable()
            print(f"freedrive enabled: {d['freedrive']['controller_name']} (restores {d['freedrive']['motion_controller_name']} on exit)",flush=True)
        except Exception:
            if freedrive.enabled:
                try:freedrive.disable_and_restore()
                except Exception as restore_error:print(f"ERROR: failed to restore controller after freedrive setup error: {restore_error}",flush=True)
            node.destroy_node();rclpy.shutdown();update_status(root,"failed","could not enable freedrive");raise
    try:
        bag_pid=bag.start()
    except Exception:
        if freedrive is not None:
            try: freedrive.disable_and_restore()
            except Exception as restore_error:
                print(f"ERROR: failed to restore controller after rosbag start error: {restore_error}",flush=True)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        update_status(root,"failed","could not start rosbag")
        raise
    start=time.monotonic(); last_progress=start
    thread=threading.Thread(target=_keyboard,args=(q,stop),daemon=True); thread.start()
    mode=f"LIVE DO{d['gripper']['output_pin']}" if execute else "DRY-RUN"
    freedrive_mode="managed freedrive" if enable_freedrive else "external freedrive"
    print(
        f"episode: {root}\n"
        f"mode: {mode} | {freedrive_mode} | rosbag pid: {bag_pid}\n"
        "keys: 0/o open, 1/c close, q finish, Esc abort",
        flush=True,
    )
    period=1/d["sampling"]["demonstration_rate_hz"]; next_sample=time.monotonic()
    events=open(root/"demonstration/events.jsonl","a",encoding="utf-8")
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node,timeout_sec=min(0.01,max(0,next_sample-time.monotonic())))
            cmd=q.get_nowait()
            while cmd:
                now=time.monotonic(); event={"event":"keyboard_"+cmd.kind,"key":cmd.key,"receipt_time_ns":node.get_clock().now().nanoseconds,"monotonic_time_ns":time.monotonic_ns(),"elapsed_sec":now-start}
                if cmd.kind=="gripper":
                    result=grip.command_semantic(cmd.semantic_state,execute=execute); event.update({"semantic_state":cmd.semantic_state,"output_pin":d["gripper"]["output_pin"],"output_value":result.output_value,"execute":execute,"service_called":result.service_called,"service_success":result.success,"reason":result.reason})
                    state_name="OPEN" if cmd.semantic_state==0 else "CLOSED"
                    outcome="OK" if result.success else "FAILED"
                    print(
                        f"[{event['elapsed_sec']:7.3f}s] gripper {state_name} "
                        f"(semantic={cmd.semantic_state}, DO{d['gripper']['output_pin']}={result.output_value:g}) "
                        f"{outcome} [{result.reason or mode}]",
                        flush=True,
                    )
                    if result.success: semantic=cmd.semantic_state
                    elif execute: failure=result.reason; stop.set()
                elif cmd.kind=="finish":
                    print(f"[{event['elapsed_sec']:7.3f}s] finish requested; finalizing episode...",flush=True)
                    stop.set()
                elif cmd.kind=="abort":
                    print(f"[{event['elapsed_sec']:7.3f}s] abort requested; preserving captured data...",flush=True)
                    aborted=True; stop.set()
                events.write(json.dumps(event)+"\n"); events.flush(); cmd=q.get_nowait()
            now=time.monotonic()
            if now>=next_sample:
                next_sample+=period; msg=joint["msg"]
                if msg is None or semantic is None: continue
                if len(msg.name)!=len(set(msg.name)): failure="duplicate joint name"; break
                idx={name:i for i,name in enumerate(msg.name)}
                if any(n not in idx for n in d["robot"]["joint_names"]): continue
                pos=np.array([msg.position[idx[n]] for n in d["robot"]["joint_names"]],float)
                vel=np.array([msg.velocity[idx[n]] if idx[n]<len(msg.velocity) else np.nan for n in d["robot"]["joint_names"]])
                try:
                    tf=tfbuf.lookup_transform(d["robot"]["base_frame"],d["robot"]["tcp_frame"],rclpy.time.Time())
                except Exception: continue
                tr=tf.transform.translation; ro=tf.transform.rotation
                T=quaternion_pose_to_matrix([tr.x,tr.y,tr.z],[ro.x,ro.y,ro.z,ro.w])
                stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
                tfstamp=tf.header.stamp.sec*1_000_000_000+tf.header.stamp.nanosec; receipt=node.get_clock().now().nanoseconds
                tcp_age_ms=max(0,(receipt-tfstamp)/1e6)
                tcp_valid=tcp_age_ms<=d["synchronization"]["max_tcp_age_ms"]
                tcp_pose=matrix_to_ur_pose(T) if tcp_valid else np.full(6,np.nan)
                out=adapter.latest_readback().value if adapter and adapter.latest_readback() else grip.output_from_semantic(semantic)
                samples.append((now-start,time.monotonic_ns(),stamp,joint["receipt"],pos,vel,tfstamp,receipt,tcp_pose,semantic,out,max(0,(receipt-stamp)/1e6),tcp_age_ms,tcp_valid))
            if now-last_progress>=1.0:
                print(
                    f"[{now-start:7.1f}s] recording | valid samples: {len(samples)}"
                    f" | gripper: {'unknown' if semantic is None else semantic}",
                    flush=True,
                )
                last_progress=now
    except KeyboardInterrupt: aborted=True
    finally:
        stop.set()
        final_open=grip.command_semantic(int(d["gripper"]["semantic_open"]),execute=execute)
        if final_open.success:
            print(
                f"final gripper OPEN (semantic={d['gripper']['semantic_open']}, "
                f"DO{d['gripper']['output_pin']}={final_open.output_value:g})",
                flush=True,
            )
            spin_for(d["gripper"]["actuation_settle_sec"])
        elif execute:
            failure=failure or f"failed to open gripper after recording: {final_open.reason}"
        events.close(); bag_code=bag.stop()
        if freedrive is not None:
            try:
                freedrive.disable_and_restore()
                print(f"freedrive disabled; restored {d['freedrive']['motion_controller_name']}",flush=True)
            except Exception as e:
                failure=failure or f"failed to restore motion controller: {e}"
                print(f"ERROR: {failure}",flush=True)
        if return_to_start and not aborted and failure is None and samples:
            try:
                from .ros_adapters.freedrive_controller import FreedriveController
                from .ros_adapters.trajectory_client import TrajectoryClient
                controller_manager=freedrive or FreedriveController(node,d["freedrive"])
                controller_manager.ensure_motion_controller()
                print(
                    "WARNING: returning to the first recorded joint position; no collision checking is performed.\n"
                    f"duration={return_to_start_duration_sec:.1f}s",
                    flush=True,
                )
                client=TrajectoryClient(node,d["ros"]["trajectory_action"],d["replay"]["controller_joint_order"],lambda _:None)
                goal=client.make_goal([return_to_start_duration_sec],[np.asarray(samples[0][4],dtype=float)])
                future=client.client.send_goal_async(goal)
                rclpy.spin_until_future_complete(node,future,timeout_sec=5)
                handle=future.result()
                if handle is None or not handle.accepted:raise RuntimeError("recording return-to-start goal rejected")
                result=handle.get_result_async();deadline=time.monotonic()+return_to_start_duration_sec*10+10
                while not result.done():
                    rclpy.spin_once(node,timeout_sec=.02)
                    if time.monotonic()>deadline:
                        handle.cancel_goal_async();raise RuntimeError("recording return-to-start result timeout")
                code=result.result().result.error_code
                if code!=0:raise RuntimeError(f"recording return-to-start failed with error_code={code}")
                return_to_start_completed=True
                print("recording return-to-start complete",flush=True)
            except Exception as e:
                failure=failure or str(e)
                print(f"ERROR: {failure}",flush=True)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    if samples:
        fields=list(zip(*samples))
        tcp_valid=np.asarray(fields[13],dtype=bool)
        np.savez_compressed(root/"demonstration/samples.npz",elapsed_sec=fields[0],monotonic_time_ns=fields[1],joint_source_time_ns=fields[2],joint_receipt_time_ns=fields[3],joint_position=np.stack(fields[4]),joint_velocity=np.stack(fields[5]),tcp_source_time_ns=fields[6],tcp_receipt_time_ns=fields[7],tcp_pose6=np.stack(fields[8]),gripper_semantic_state=fields[9],digital_output_value=fields[10],state_age_ms=fields[11],tcp_age_ms=fields[12],tcp_valid=tcp_valid)
        from .core.quality_metrics import age_metrics
        if not tcp_valid.any():
            failure="no TCP samples passed the configured stale threshold"
            val={"valid":False,"errors":[failure],"sample_count":0,"duration_sec":0.0,"gripper_transitions":[]}
        else:
            valid_indices=np.flatnonzero(tcp_valid)
            t=np.asarray(fields[0])[valid_indices]; t=t-t[0]
            joint_position=np.stack(fields[4])[valid_indices];joint_velocity=np.stack(fields[5])[valid_indices]
            tcp_pose=np.stack(fields[8])[valid_indices];gripper_state=np.asarray(fields[9])[valid_indices];digital_output=np.asarray(fields[10])[valid_indices]
            np.savez_compressed(root/"demonstration/trajectory.npz",time_from_start_sec=t,joint_position=joint_position,joint_velocity=joint_velocity,tcp_pose6=tcp_pose,gripper_semantic_state=gripper_state,digital_output_value=digital_output)
            val=validate_arrays(t,joint_position,gripper_state,tcp_pose,d["replay"]["max_joint_velocity_rad_s"],d["replay"]["max_joint_acceleration_rad_s2"])
        val.update(age_metrics(fields[12],"tcp_age"))
        val["tcp_valid_sample_count"]=int(tcp_valid.sum())
        val["tcp_stale_sample_count"]=int((~tcp_valid).sum())
        val["return_to_start_requested"]=bool(return_to_start)
        val["return_to_start_completed"]=bool(return_to_start_completed)
        val["raw_bag_present"]=bag.metadata_exists(); val["bag_exit_code"]=bag_code
        (root/"demonstration/validation.json").write_text(json.dumps(val,indent=2)+"\n")
    else: failure=failure or "no valid samples (joint, TF, and known initial gripper are required)"
    if failure:
        update_status(root,"failed",failure)
        print(f"FAILED: {failure}\nepisode preserved at: {root}",flush=True)
        return 2
    if aborted:
        update_status(root,"aborted","keyboard/Ctrl+C")
        print(f"ABORTED: captured data preserved at {root}",flush=True)
        return 130
    update_status(root,"demo_recorded")
    update_status(root,"demo_validated" if val["valid"] else "demo_recorded",None if val["valid"] else "; ".join(val["errors"]))
    print(
        f"{'VALID' if val['valid'] else 'INVALID'}: samples={val['sample_count']} "
        f"duration={val['duration_sec']:.3f}s raw_bag={val['raw_bag_present']}\n"
        f"episode saved at: {root}",
        flush=True,
    )
    return 0 if val["valid"] else 2

def main():
    from .collector_cli import main as cli
    cli(["record-demo",*sys.argv[1:]])

if __name__ == "__main__":
    main()
