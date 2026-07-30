"""Publish raw GRBG frames from the deployed oCam as ROS images."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import BinaryIO, Optional
import cv2
from cv_bridge import CvBridge
import numpy as np

DEFAULT_DEVICE="/dev/v4l/by-id/usb-WITHROBOT_Inc._oCam-1CGN-U-T2_SN_3AA01020-video-index0"

def debayer_grbg(frame:np.ndarray,width:int,height:int)->np.ndarray:
    raw=np.asarray(frame)
    if raw.size!=width*height:raise ValueError(f"expected {width*height} Bayer bytes, received {raw.size}")
    raw=raw.reshape(height,width).astype(np.uint8,copy=False)
    # Preserve the conversion used by the known-working legacy publisher.
    return np.ascontiguousarray(cv2.cvtColor(raw,cv2.COLOR_BayerGR2RGB))

class OCamPublisher:
    def __init__(self,node,device,width,height,fps,frame_id):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        self.node=node; self.device=str(Path(device).resolve(strict=True))
        self.width=width; self.height=height; self.fps=fps; self.frame_id=frame_id
        if width<=0 or height<=0 or fps<=0:raise RuntimeError("width, height, and fps must be positive")
        if shutil.which("v4l2-ctl") is None:raise RuntimeError("v4l2-ctl not found; install v4l-utils")
        self.frame_size=width*height; self.bridge=CvBridge()
        self.publisher=node.create_publisher(Image,"image_raw",qos_profile_sensor_data)
        self.process:Optional[subprocess.Popen[bytes]]=None; self.stderr_thread=None
        self.shutting_down=False; self.frame_count=0; self.last_warning=0.0; self.last_restart=0.0
        self._configure(); self._start()
        self.timer=node.create_timer(1.0/fps,self._capture)
        node.get_logger().info(
            f"oCam started: device={device} -> {self.device}, "
            f"size={width}x{height}, fps={fps:g}, topic=/wrist_camera/image_raw"
        )
    def _run(self,args):
        command=["v4l2-ctl","-d",self.device,*args]
        self.node.get_logger().info("Running: "+" ".join(command))
        return subprocess.run(command,capture_output=True,text=True,check=False)
    def _configure(self):
        result=self._run(["--set-fmt-video",f"width={self.width},height={self.height},pixelformat=GRBG","--set-parm",str(int(round(self.fps)))])
        if result.returncode!=0:raise RuntimeError("failed to configure oCam: "+result.stderr.strip())
        verify=self._run(["--get-fmt-video","--get-parm"])
        if verify.returncode==0:self.node.get_logger().info("Current oCam configuration:\n"+verify.stdout.strip())
        else:self.node.get_logger().warning("Could not verify oCam: "+verify.stderr.strip())
    def _start(self):
        self._stop()
        command=["v4l2-ctl","-d",self.device,"--stream-mmap=4","--stream-to=/dev/stdout"]
        self.node.get_logger().info("Starting: "+" ".join(command))
        self.process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        if self.process.stdout is None:self._stop();raise RuntimeError("failed to create v4l2-ctl stdout pipe")
        self.stderr_thread=threading.Thread(target=self._read_stderr,daemon=True);self.stderr_thread.start()
        time.sleep(.1)
        if self.process.poll() is not None:
            code=self.process.returncode;self._stop();raise RuntimeError(f"v4l2-ctl exited immediately with code {code}")
    def _stop(self):
        p=self.process;self.process=None
        if p is None:return
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=1)
        if p.stdout:p.stdout.close()
        if p.stderr:p.stderr.close()
    def _read_stderr(self):
        p=self.process
        if p is None or p.stderr is None:return
        try:
            while not self.shutting_down:
                line=p.stderr.readline()
                if not line:return
                text=line.decode(errors="replace").strip()
                if text and set(text)!={"<"}:self.node.get_logger().info("[v4l2-ctl] "+text)
        except (OSError,ValueError):return
    @staticmethod
    def _read_exactly(stream:BinaryIO,size:int)->Optional[bytes]:
        data=bytearray()
        while len(data)<size:
            chunk=stream.read(size-len(data))
            if not chunk:return None
            data.extend(chunk)
        return bytes(data)
    def _capture(self):
        if self.shutting_down:return
        p=self.process
        if p is None or p.stdout is None or p.poll() is not None:
            self._warn("oCam stream is not running");self._restart();return
        data=self._read_exactly(p.stdout,self.frame_size)
        if data is None:self._warn("failed to read complete oCam frame");self._restart();return
        try:
            raw=np.frombuffer(data,np.uint8).reshape(self.height,self.width)
            bgr=debayer_grbg(raw,self.width,self.height)
            msg=self.bridge.cv2_to_imgmsg(bgr,encoding="bgr8")
        except (ValueError,cv2.error,RuntimeError) as e:self._warn(f"frame conversion failed: {e}");return
        msg.header.stamp=self.node.get_clock().now().to_msg();msg.header.frame_id=self.frame_id
        self.publisher.publish(msg);self.frame_count+=1
        if self.frame_count==1:self.node.get_logger().info(f"first frame: shape={raw.shape}, min={raw.min()}, max={raw.max()}")
        elif self.frame_count%600==0:self.node.get_logger().info(f"published {self.frame_count} frames")
    def _restart(self):
        now=time.monotonic()
        if now-self.last_restart<2:return
        self.last_restart=now
        try:self.node.get_logger().warning("restarting oCam stream");self._configure();self._start()
        except (OSError,RuntimeError) as e:self._warn(f"restart failed: {e}")
    def _warn(self,message):
        now=time.monotonic()
        if now-self.last_warning>=1:self.last_warning=now;self.node.get_logger().warning(message)
    def close(self):
        self.shutting_down=True
        if hasattr(self,"timer"):self.timer.cancel()
        self._stop();self.node.get_logger().info(f"oCam stopped after {self.frame_count} frames")

def main(argv=None):
    p=argparse.ArgumentParser(description="oCam GRBG wrist camera ROS 2 publisher")
    p.add_argument("--device",default=DEFAULT_DEVICE);p.add_argument("--width",type=int,default=1280)
    p.add_argument("--height",type=int,default=800);p.add_argument("--fps",type=float,default=60.0)
    p.add_argument("--frame-id",default="ocam_optical_frame")
    args,ros_args=p.parse_known_args(argv)
    import rclpy
    from rclpy.node import Node
    rclpy.init(args=ros_args);node=Node("ocam_publisher",namespace="/wrist_camera");camera=None
    try:camera=OCamPublisher(node,args.device,args.width,args.height,args.fps,args.frame_id);rclpy.spin(node)
    except KeyboardInterrupt:pass
    except (OSError,RuntimeError) as e:node.get_logger().fatal(str(e))
    finally:
        if camera:camera.close()
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
