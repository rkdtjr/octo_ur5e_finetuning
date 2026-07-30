from __future__ import annotations
import json,select,sys,termios,threading,time,tty
import numpy as np
from .core.episode import create_episode,update_status,utc_now
from .core.gripper import GripperController
from .core.keyboard_commands import KeyboardCommandQueue
from .core.trajectory import validate_arrays
from .core.transforms import matrix_to_ur_pose,quaternion_pose_to_matrix
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

def run_recording(config,instruction,execute=False,initial_gripper=None):
    checks=run_preflight(config,execute,False)
    for c in checks: print(f"{'OK' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    if not preflight_ok(checks): return 2
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer,TransformListener
    except ImportError as e: raise RuntimeError(f"ROS Python packages unavailable: {e}") from e
    root=create_episode(config,instruction); update_status(root,"recording_demo")
    d=config.data; bag=RosbagRecorder(root/"demonstration/rosbag2",d["raw_topics"]["demonstration"],d["storage"]["rosbag_storage_id"])
    rclpy.init(); node=Node("octo_record_demonstration")
    joint={"msg":None,"receipt":0}; samples=[]; q=KeyboardCommandQueue(d["keyboard"]); stop=threading.Event(); aborted=False; failure=None
    def joint_cb(msg): joint.update(msg=msg,receipt=node.get_clock().now().nanoseconds)
    sub=node.create_subscription(JointState,d["ros"]["joint_state_topic"],joint_cb,50)
    tfbuf=Buffer(); listener=TransformListener(tfbuf,node)
    adapter=None
    if execute:
        from .ros_adapters.digital_output_gripper import DigitalOutputGripperAdapter
        adapter=DigitalOutputGripperAdapter(node,{**d["gripper"],**d["ros"]})
    grip=GripperController(d["gripper"],(lambda value:adapter.send(value,d["gripper"]["command_timeout_sec"])) if adapter else None)
    semantic=None
    if initial_gripper is not None: semantic=0 if initial_gripper=="open" else 1
    elif adapter is not None:
        deadline=time.monotonic()+d["gripper"]["confirmation_timeout_sec"]
        while adapter.latest_readback() is None and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.05)
        if adapter.latest_readback() is not None:
            try: semantic=grip.semantic_from_output(adapter.latest_readback().value)
            except ValueError: pass
    bag.start(); start=time.monotonic(); thread=threading.Thread(target=_keyboard,args=(q,stop),daemon=True); thread.start()
    print(f"episode: {root}\nkeys: 0/o open, 1/c close, q finish, Esc abort")
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
                    if result.success: semantic=cmd.semantic_state
                    elif execute: failure=result.reason; stop.set()
                elif cmd.kind=="finish": stop.set()
                elif cmd.kind=="abort": aborted=True; stop.set()
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
                out=adapter.latest_readback().value if adapter and adapter.latest_readback() else grip.output_from_semantic(semantic)
                samples.append((now-start,time.monotonic_ns(),stamp,joint["receipt"],pos,vel,tfstamp,receipt,matrix_to_ur_pose(T),semantic,out,max(0,(receipt-stamp)/1e6),max(0,(receipt-tfstamp)/1e6)))
    except KeyboardInterrupt: aborted=True
    finally:
        stop.set(); events.close(); bag_code=bag.stop(); node.destroy_node(); rclpy.shutdown()
    if samples:
        fields=list(zip(*samples))
        np.savez_compressed(root/"demonstration/samples.npz",elapsed_sec=fields[0],monotonic_time_ns=fields[1],joint_source_time_ns=fields[2],joint_receipt_time_ns=fields[3],joint_position=np.stack(fields[4]),joint_velocity=np.stack(fields[5]),tcp_source_time_ns=fields[6],tcp_receipt_time_ns=fields[7],tcp_pose6=np.stack(fields[8]),gripper_semantic_state=fields[9],digital_output_value=fields[10],state_age_ms=fields[11],tf_age_ms=fields[12])
        t=np.asarray(fields[0]); t=t-t[0]
        np.savez_compressed(root/"demonstration/trajectory.npz",time_from_start_sec=t,joint_position=np.stack(fields[4]),joint_velocity=np.stack(fields[5]),tcp_pose6=np.stack(fields[8]),gripper_semantic_state=fields[9],digital_output_value=fields[10])
        val=validate_arrays(t,np.stack(fields[4]),fields[9],np.stack(fields[8]),d["replay"]["max_joint_velocity_rad_s"],d["replay"]["max_joint_acceleration_rad_s2"])
        val["raw_bag_present"]=bag.metadata_exists(); val["bag_exit_code"]=bag_code
        (root/"demonstration/validation.json").write_text(json.dumps(val,indent=2)+"\n")
    else: failure=failure or "no valid samples (joint, TF, and known initial gripper are required)"
    if failure:update_status(root,"failed",failure); print(f"failed: {failure}"); return 2
    if aborted:update_status(root,"aborted","keyboard/Ctrl+C"); return 130
    update_status(root,"demo_recorded")
    update_status(root,"demo_validated" if val["valid"] else "demo_recorded",None if val["valid"] else "; ".join(val["errors"]))
    return 0 if val["valid"] else 2

def main():
    from .collector_cli import main as cli
    cli(["record-demo",*sys.argv[1:]])

if __name__ == "__main__":
    main()
