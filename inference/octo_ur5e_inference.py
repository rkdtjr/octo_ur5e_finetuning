"""Guarded live Octo inference and optional MoveIt-IK robot execution."""
from __future__ import annotations
import argparse,base64,json,select,subprocess,time
from collections import deque
from pathlib import Path
import cv2,numpy as np
from inference.safety import GripperDebouncer,limit_action,validate_target_position
from octo_ur5e_collector.core.config import load_config
from octo_ur5e_collector.core.transforms import apply_relative_pose_action,matrix_to_quaternion_pose,quaternion_pose_to_matrix,relative_pose_action

DEFAULT_CHECKPOINT="runs/octo_ur5e/pick/ur5e_pick_resume_1k_to_10k_20260801_systemd_20260801_014940"

def preprocess_primary(bgr):
    h,w=bgr.shape[:2];side=min(h,w);y=(h-side)//2;x=(w-side)//2
    return cv2.resize(bgr[y:y+side,x:x+side],(256,256),interpolation=cv2.INTER_AREA)

def preprocess_wrist(bgr):
    h,w=bgr.shape[:2];side=min(h,w);y=(h-side)//2;x=(w-side)//2
    return cv2.resize(bgr[y:y+side,x:x+side],(128,128),interpolation=cv2.INTER_AREA)

def encode_frame(bgr):
    ok,value=cv2.imencode(".jpg",bgr,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok:raise RuntimeError("JPEG encode failed")
    return base64.b64encode(value).decode()

def select_frame_pair(history,history_sec=.1,tolerance_sec=.06):
    """Select the latest frame and the frame nearest one training period ago."""
    if len(history)<2:return None
    latest=history[-1];target_ns=latest[0]-int(history_sec*1e9)
    previous=min(list(history)[:-1],key=lambda item:abs(item[0]-target_ns))
    gap_sec=(latest[0]-previous[0])/1e9
    if gap_sec<=0 or abs(gap_sec-history_sec)>tolerance_sec:return None
    return previous[1],latest[1],gap_sec

def select_synchronized_pair(primary_history,wrist_history,history_sec=.1,history_tolerance_sec=.06,sync_tolerance_sec=.04):
    if len(primary_history)<2 or len(wrist_history)<2:return None
    latest=primary_history[-1];target_ns=latest[0]-int(history_sec*1e9)
    previous=min(list(primary_history)[:-1],key=lambda item:abs(item[0]-target_ns))
    gap_sec=(latest[0]-previous[0])/1e9
    if gap_sec<=0 or abs(gap_sec-history_sec)>history_tolerance_sec:return None
    wrist=[];errors=[]
    for stamp,frame in (previous,latest):
        match=min(wrist_history,key=lambda item:abs(item[0]-stamp))
        error_sec=abs(match[0]-stamp)/1e9
        if error_sec>sync_tolerance_sec:return None
        wrist.append(match[1]);errors.append(error_sec)
    return [previous[1],latest[1]],wrist,gap_sec,max(errors)

class Worker:
    def __init__(self,checkpoint,step,python,use_wrist=False):
        command=[python,"-m","inference.model_worker","--checkpoint",str(checkpoint),"--step",str(step)]
        if use_wrist:command.append("--use-wrist")
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        ready=json.loads(self.process.stdout.readline())
        if not ready.get("ready"):raise RuntimeError(f"model worker failed: {ready}")
    def infer(self,frames,instruction,wrist_frames=None,on_wait=None,timeout_sec=60.):
        payload={"primary_jpeg_b64":[encode_frame(x) for x in frames],"instruction":instruction}
        if wrist_frames is not None:payload["wrist_jpeg_b64"]=[encode_frame(x) for x in wrist_frames]
        self.process.stdin.write(json.dumps(payload)+"\n");self.process.stdin.flush();deadline=time.monotonic()+timeout_sec
        while not select.select([self.process.stdout],[],[],0)[0]:
            if self.process.poll() is not None:raise RuntimeError("model worker exited during inference")
            if time.monotonic()>deadline:raise RuntimeError("model inference timed out")
            if on_wait is not None:on_wait()
            else:time.sleep(.01)
        result=json.loads(self.process.stdout.readline())
        if not result.get("ok"):raise RuntimeError(result.get("error","worker failed"))
        chunk=np.asarray(result["action_chunk"],float)
        if chunk.ndim!=2 or chunk.shape[1]!=7:raise RuntimeError(f"invalid model action chunk shape: {chunk.shape}")
        return chunk
    def close(self):
        if self.process.poll() is None:self.process.terminate()

def run(args):
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.srv import GetPositionIK
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image,JointState
    from tf2_ros import Buffer,TransformListener
    from octo_ur5e_collector.ros_adapters.camera_video_recorder import image_message_to_bgr
    from octo_ur5e_collector.ros_adapters.preflight import run_preflight,preflight_ok
    config=load_config(args.config);d=config.data
    if args.execute and not args.confirm_real_robot:raise ValueError("--execute requires --confirm-real-robot")
    checks=run_preflight(config,execute=args.execute,replay=False,motion=args.execute)
    for check in checks:print(f"{'OK' if check.ok else 'FAIL'} {check.name}: {check.detail}",flush=True)
    if not preflight_ok(checks):raise RuntimeError("required robot preflight failed")
    rclpy.init();node=Node("octo_ur5e_policy");bridge=CvBridge();latest={"frame":None,"frame_time":0.,"wrist_frame":None,"wrist_time":0.,"joints":None,"joint_time":0.};frame_history=deque(maxlen=60);wrist_history=deque(maxlen=60)
    def image_cb(msg):
        try:
            frame=preprocess_primary(image_message_to_bgr(bridge,msg));stamp_ns=int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nanosec)
            if stamp_ns<=0:stamp_ns=time.monotonic_ns()
            latest["frame"]=frame;latest["frame_time"]=time.monotonic()
            if not frame_history or stamp_ns>frame_history[-1][0]:frame_history.append((stamp_ns,frame))
        except Exception as error:node.get_logger().error(f"camera frame rejected: {error}")
    def wrist_cb(msg):
        try:
            frame=preprocess_wrist(image_message_to_bgr(bridge,msg));stamp_ns=int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nanosec)
            if stamp_ns<=0:stamp_ns=time.monotonic_ns()
            latest["wrist_frame"]=frame;latest["wrist_time"]=time.monotonic()
            if not wrist_history or stamp_ns>wrist_history[-1][0]:wrist_history.append((stamp_ns,frame))
        except Exception as error:node.get_logger().error(f"wrist camera frame rejected: {error}")
    def joint_cb(msg):
        index={name:i for i,name in enumerate(msg.name)}
        if all(name in index for name in d["robot"]["joint_names"]):
            latest["joints"]=np.array([msg.position[index[name]] for name in d["robot"]["joint_names"]]);latest["joint_time"]=time.monotonic()
    node.create_subscription(Image,d["cameras"][0]["image_topic"],image_cb,qos_profile_sensor_data)
    if args.use_wrist:
        wrist_camera=next((camera for camera in d["cameras"] if camera.get("logical_name")=="wrist"),None)
        if wrist_camera is None:raise RuntimeError("--use-wrist requested but no wrist camera is configured")
        node.create_subscription(Image,wrist_camera["image_topic"],wrist_cb,qos_profile_sensor_data)
    node.create_subscription(JointState,d["ros"]["joint_state_topic"],joint_cb,20)
    tf_buffer=Buffer();listener=TransformListener(tf_buffer,node)
    ik=node.create_client(GetPositionIK,"/compute_ik") if args.execute else None
    if args.execute and not ik.wait_for_service(timeout_sec=3):raise RuntimeError("MoveIt /compute_ik unavailable; start move_group before --execute")
    startup_deadline=time.monotonic()+args.startup_timeout_sec
    while rclpy.ok():
        data_ready=latest["frame"] is not None and latest["joints"] is not None and (not args.use_wrist or latest["wrist_frame"] is not None)
        tf_ready=tf_buffer.can_transform(d["robot"]["base_frame"],d["robot"]["tcp_frame"],rclpy.time.Time())
        if data_ready and tf_ready:break
        if time.monotonic()>startup_deadline:
            missing=[]
            if latest["frame"] is None:missing.append(d["cameras"][0]["image_topic"])
            if args.use_wrist and latest["wrist_frame"] is None:missing.append(wrist_camera["image_topic"])
            if latest["joints"] is None:missing.append(d["ros"]["joint_state_topic"])
            if not tf_ready:missing.append(f'TF {d["robot"]["base_frame"]} -> {d["robot"]["tcp_frame"]}')
            raise RuntimeError("startup timed out waiting for: "+", ".join(missing))
        rclpy.spin_once(node,timeout_sec=.05)
    worker=Worker(Path(args.checkpoint).resolve(),args.step,args.octo_python,args.use_wrist);grip=None;gripper=None;client=None
    gripper_filter=GripperDebouncer(args.gripper_close_threshold,args.gripper_open_threshold,args.gripper_close_debounce_steps,args.gripper_open_debounce_steps)
    if args.execute:
        from octo_ur5e_collector.ros_adapters.digital_output_gripper import DigitalOutputGripperAdapter
        from octo_ur5e_collector.core.gripper import GripperController
        from octo_ur5e_collector.ros_adapters.trajectory_client import TrajectoryClient
        adapter=DigitalOutputGripperAdapter(node,{**d["gripper"],**d["ros"]});gripper=GripperController(d["gripper"],lambda value:adapter.send(value,d["gripper"]["command_timeout_sec"]));client=TrajectoryClient(node,d["ros"]["trajectory_action"],d["robot"]["joint_names"])
        if not client.client.wait_for_server(timeout_sec=3):raise RuntimeError("trajectory action unavailable; verify scaled_joint_trajectory_controller is running")
        opened=gripper.command_semantic(d["gripper"]["semantic_open"],execute=True)
        if not opened.success:raise RuntimeError("failed to initialize gripper open")
        grip=d["gripper"]["semantic_open"];deadline=time.monotonic()+d["gripper"]["actuation_settle_sec"]
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
    def current_tcp():
        if not tf_buffer.can_transform(d["robot"]["base_frame"],d["robot"]["tcp_frame"],rclpy.time.Time()):raise RuntimeError("TCP transform became unavailable")
        transform=tf_buffer.lookup_transform(d["robot"]["base_frame"],d["robot"]["tcp_frame"],rclpy.time.Time());tf_stamp=rclpy.time.Time.from_msg(transform.header.stamp)
        tf_age=(node.get_clock().now()-tf_stamp).nanoseconds/1e9
        if tf_age<0 or tf_age>args.max_tf_age_sec:raise RuntimeError(f"TCP transform stale: age={tf_age:.3f}s")
        t=transform.transform.translation;q=transform.transform.rotation
        return quaternion_pose_to_matrix([t.x,t.y,t.z],[q.x,q.y,q.z,q.w])
    log_path=Path(args.log_file).resolve() if args.log_file else Path("runs/policy_logs")/f"policy_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()%1_000_000_000:09d}.jsonl"
    log_path.parent.mkdir(parents=True,exist_ok=True);log_handle=log_path.open("x",encoding="utf-8");print(f"policy_log: {log_path.resolve()}",flush=True)
    def emit(record):
        line=json.dumps(record);print(line,flush=True);log_handle.write(line+"\n");log_handle.flush()
    emit({"event":"run_start","mode":"EXECUTE" if args.execute else "DRY_RUN","current_step":0,"max_steps":args.max_steps,"action_interval_sec":args.command_duration_sec,"action_chunk_steps":args.action_chunk_steps,"close_threshold":args.gripper_close_threshold,"open_threshold":args.gripper_open_threshold,"close_debounce_steps":args.gripper_close_debounce_steps,"open_debounce_steps":args.gripper_open_debounce_steps})
    stop_reason="unknown";step=0
    try:
        while rclpy.ok() and step<args.max_steps:
            fresh_deadline=time.monotonic()+args.startup_timeout_sec;frame_pair=None;wrist_frames=None;camera_sync_sec=None
            while rclpy.ok() and (
                latest["frame"] is None or latest["joints"] is None
                or time.monotonic()-latest["frame_time"]>.5
                or (args.use_wrist and time.monotonic()-latest["wrist_time"]>.5)
                or time.monotonic()-latest["joint_time"]>.5
                or frame_pair is None
            ):
                if time.monotonic()>fresh_deadline:raise RuntimeError("timed out waiting for fresh synchronized camera and joint states")
                rclpy.spin_once(node,timeout_sec=.05)
                if args.use_wrist:
                    synchronized=select_synchronized_pair(frame_history,wrist_history,args.observation_history_sec,args.observation_history_tolerance_sec,args.camera_sync_tolerance_sec)
                    if synchronized is not None:
                        frames,wrist_frames,history_gap_sec,camera_sync_sec=synchronized;frame_pair=frames
                else:frame_pair=select_frame_pair(frame_history,args.observation_history_sec,args.observation_history_tolerance_sec)
            if not args.use_wrist:
                previous_frame,current_frame,history_gap_sec=frame_pair;frames=[previous_frame.copy(),current_frame.copy()]
            else:
                frames=[frame.copy() for frame in frames];wrist_frames=[frame.copy() for frame in wrist_frames]
            predicted=worker.infer(frames,args.instruction,wrist_frames=wrist_frames,on_wait=lambda:rclpy.spin_once(node,timeout_sec=.01))
            count=min(len(predicted),args.action_chunk_steps,args.max_steps-step);raw_chunk=predicted[:count];safe_chunk=[limit_action(raw,args.max_translation_m,args.max_rotation_rad) for raw in raw_chunk]
            before=current_tcp();target=before.copy();targets=[]
            for safe in safe_chunk:
                target=apply_relative_pose_action(target,safe[:6],"tool");validate_target_position(target[:3,3]);targets.append(target.copy())
            if args.execute:
                joint_seed=latest["joints"].copy();joint_targets=[]
                for target in targets:
                    position,quat=matrix_to_quaternion_pose(target);request=GetPositionIK.Request();request.ik_request.group_name=args.move_group;request.ik_request.avoid_collisions=True;request.ik_request.timeout.sec=1
                    request.ik_request.robot_state.joint_state.name=list(d["robot"]["joint_names"]);request.ik_request.robot_state.joint_state.position=joint_seed.tolist()
                    request.ik_request.pose_stamped=PoseStamped();request.ik_request.pose_stamped.header.frame_id=d["robot"]["base_frame"];request.ik_request.pose_stamped.header.stamp=node.get_clock().now().to_msg();pose=request.ik_request.pose_stamped.pose;pose.position.x,pose.position.y,pose.position.z=map(float,position);pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,quat)
                    future=ik.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=2)
                    if not future.done() or future.result().error_code.val!=1:raise RuntimeError("collision-aware IK failed")
                    solution=future.result().solution.joint_state;idx={name:i for i,name in enumerate(solution.name)};goal_q=np.array([solution.position[idx[name]] for name in d["robot"]["joint_names"]])
                    if np.max(np.abs(goal_q-joint_seed))>args.max_joint_step_rad:raise RuntimeError("IK joint step exceeds safety limit")
                    joint_targets.append(goal_q);joint_seed=goal_q
                times=[args.command_duration_sec*(i+1) for i in range(count)];captured=[None]*count;captured_progress=[None]*count;controller_progress=0.
                def feedback_callback(message):
                    nonlocal controller_progress
                    duration=message.feedback.desired.time_from_start;controller_progress=float(duration.sec)+float(duration.nanosec)/1e9
                    for index,point_time in enumerate(times):
                        if captured[index] is None and controller_progress>=point_time:
                            try:captured[index]=current_tcp();captured_progress[index]=controller_progress
                            except RuntimeError:pass
                send_wall=time.monotonic();result=client.client.send_goal_async(client.make_goal(times,joint_targets),feedback_callback=feedback_callback);rclpy.spin_until_future_complete(node,result,timeout_sec=2);handle=result.result()
                if handle is None or not handle.accepted:raise RuntimeError("trajectory goal rejected")
                completed=handle.get_result_async();next_gripper=0;grip_states=[None]*count;deadline=time.monotonic()+args.command_duration_sec*count*5+2
                while rclpy.ok() and not completed.done():
                    if time.monotonic()>deadline:raise RuntimeError("trajectory result timeout")
                    rclpy.spin_once(node,timeout_sec=.01)
                    while next_gripper<count and controller_progress>=times[next_gripper]:
                        desired=gripper_filter.update(safe_chunk[next_gripper][6],grip)
                        if desired!=grip:
                            command=gripper.command_semantic(desired,execute=True)
                            if not command.success:raise RuntimeError("gripper command failed")
                            grip=desired
                        grip_states[next_gripper]=grip
                        next_gripper+=1
                if not completed.done() or completed.result().result.error_code!=0:raise RuntimeError("trajectory execution failed")
                while next_gripper<count:
                    desired=gripper_filter.update(safe_chunk[next_gripper][6],grip)
                    if desired!=grip:
                        command=gripper.command_semantic(desired,execute=True)
                        if not command.success:raise RuntimeError("gripper command failed")
                        grip=desired
                    grip_states[next_gripper]=grip;next_gripper+=1
                result_wall=time.monotonic();captured[-1]=current_tcp();captured_progress[-1]=max(controller_progress,times[-1]);observed=before
                for chunk_index,(raw,safe,target,actual) in enumerate(zip(raw_chunk,safe_chunk,targets,captured)):
                    actual_delta=relative_pose_action(observed,actual,"tool") if actual is not None else None
                    if actual is not None:observed=actual
                    command_norm=float(np.linalg.norm(safe[:3]));actual_norm=float(np.linalg.norm(actual_delta[:3])) if actual_delta is not None else None
                    ratio=actual_norm/command_norm if actual_norm is not None and command_norm>1e-12 else None
                    pose_error=relative_pose_action(actual,target,"tool") if actual is not None else None
                    reached=bool(np.linalg.norm(pose_error[:3])<=args.goal_translation_tolerance_m and np.linalg.norm(pose_error[3:])<=args.goal_rotation_tolerance_rad) if pose_error is not None else None
                    emit({"event":"step","step":step+chunk_index,"current_step":step+chunk_index+1,"max_steps":args.max_steps,"stop_reason":None,"chunk_index":chunk_index,"mode":"EXECUTE","observation_history_ms":history_gap_sec*1000,"action_interval_sec":args.command_duration_sec,"controller_time_at_measurement_sec":captured_progress[chunk_index],"trajectory_result_wall_time_sec":result_wall-send_wall,"next_goal_sent_before_previous_result":False,"translation_clamp_m":args.max_translation_m,"policy_action":raw.tolist(),"clamped_command":safe.tolist(),"command_translation_norm_m":command_norm,"actual_tcp_delta":actual_delta.tolist() if actual_delta is not None else None,"actual_translation_norm_m":actual_norm,"actual_to_command_ratio":ratio,"target_pose_error":pose_error.tolist() if pose_error is not None else None,"target_reached":reached,"gripper_policy_value":float(safe[6]),"gripper_semantic_state":int(grip_states[chunk_index]),"gripper_hysteresis":{"close_threshold":args.gripper_close_threshold,"open_threshold":args.gripper_open_threshold,"close_debounce_steps":args.gripper_close_debounce_steps,"open_debounce_steps":args.gripper_open_debounce_steps},"target_xyz":target[:3,3].tolist(),"actual_xyz":actual[:3,3].tolist() if actual is not None else None})
            else:
                for chunk_index,(raw,safe,target) in enumerate(zip(raw_chunk,safe_chunk,targets)):
                    desired=gripper_filter.update(safe[6],grip);grip=desired
                    emit({"event":"step","step":step+chunk_index,"current_step":step+chunk_index+1,"max_steps":args.max_steps,"stop_reason":None,"chunk_index":chunk_index,"mode":"DRY_RUN","observation_history_ms":history_gap_sec*1000,"action_interval_sec":args.command_duration_sec,"next_goal_sent_before_previous_result":False,"translation_clamp_m":args.max_translation_m,"policy_action":raw.tolist(),"clamped_command":safe.tolist(),"command_translation_norm_m":float(np.linalg.norm(safe[:3])),"actual_tcp_delta":None,"actual_translation_norm_m":None,"actual_to_command_ratio":None,"target_reached":None,"gripper_policy_value":float(safe[6]),"gripper_semantic_state":int(grip),"gripper_hysteresis":{"close_threshold":args.gripper_close_threshold,"open_threshold":args.gripper_open_threshold,"close_debounce_steps":args.gripper_close_debounce_steps,"open_debounce_steps":args.gripper_open_debounce_steps},"target_xyz":target[:3,3].tolist()})
            step+=count
        stop_reason="max_steps_reached" if step>=args.max_steps else "ros_shutdown"
        emit({"event":"run_stop","stop_reason":stop_reason,"current_step":step,"max_steps":args.max_steps})
    except KeyboardInterrupt:
        stop_reason="keyboard_interrupt";emit({"event":"run_stop","stop_reason":stop_reason,"current_step":step,"max_steps":args.max_steps});raise
    except Exception as error:
        stop_reason=f"error:{type(error).__name__}";emit({"event":"run_stop","stop_reason":stop_reason,"error":str(error),"current_step":step,"max_steps":args.max_steps});raise
    finally:
        log_handle.close();worker.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

def main(argv=None):
    parser=argparse.ArgumentParser(prog="octo-policy");parser.add_argument("--checkpoint",default=DEFAULT_CHECKPOINT);parser.add_argument("--step",type=int,default=10000);parser.add_argument("--config",default="collector/config/collector.yaml");parser.add_argument("--instruction",default="pick up the blue object");parser.add_argument("--octo-python",default="/home/sixr/miniconda3/envs/octo_env/bin/python");parser.add_argument("--log-file");parser.add_argument("--use-wrist",action="store_true");parser.add_argument("--camera-sync-tolerance-sec",type=float,default=.04);parser.add_argument("--rate-hz",type=float,default=2.0,help=argparse.SUPPRESS);parser.add_argument("--max-steps",type=int,default=20);parser.add_argument("--action-chunk-steps",type=int,default=4);parser.add_argument("--observation-history-sec",type=float,default=.1);parser.add_argument("--observation-history-tolerance-sec",type=float,default=.03);parser.add_argument("--max-translation-m",type=float,default=.005);parser.add_argument("--max-rotation-rad",type=float,default=.02);parser.add_argument("--max-joint-step-rad",type=float,default=.08);parser.add_argument("--command-duration-sec",type=float,default=.1);parser.add_argument("--goal-translation-tolerance-m",type=float,default=.001);parser.add_argument("--goal-rotation-tolerance-rad",type=float,default=.01);parser.add_argument("--gripper-close-threshold",type=float,default=.9);parser.add_argument("--gripper-open-threshold",type=float,default=.1);parser.add_argument("--gripper-close-debounce-steps",type=int,default=3);parser.add_argument("--gripper-open-debounce-steps",type=int,default=3);parser.add_argument("--startup-timeout-sec",type=float,default=10.0);parser.add_argument("--max-tf-age-sec",type=float,default=.5);parser.add_argument("--move-group",default="ur_manipulator");parser.add_argument("--execute",action="store_true");parser.add_argument("--confirm-real-robot",action="store_true")
    args=parser.parse_args(argv)
    positive=(args.rate_hz,args.max_steps,args.action_chunk_steps,args.gripper_close_debounce_steps,args.gripper_open_debounce_steps,args.observation_history_sec,args.observation_history_tolerance_sec,args.camera_sync_tolerance_sec,args.max_translation_m,args.max_rotation_rad,args.max_joint_step_rad,args.command_duration_sec,args.goal_translation_tolerance_m,args.goal_rotation_tolerance_rad,args.startup_timeout_sec,args.max_tf_age_sec)
    if any(value<=0 for value in positive):parser.error("rate, step count, timeouts, and safety limits must be positive")
    if not 0<=args.gripper_open_threshold<args.gripper_close_threshold<=1:parser.error("gripper thresholds must satisfy 0 <= open < close <= 1")
    try:run(args)
    except (ValueError,RuntimeError,FileNotFoundError) as error:parser.error(str(error))

if __name__=="__main__":main()
