from __future__ import annotations
import time
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from ..core.video_recording import H264VideoWriter

class CameraVideoRecorder:
    def __init__(self,node,name,config,root,capture_fps,queue_size,container):
        self.node=node;self.name=name;self.c=config;self.bridge=CvBridge();self.warned=0
        self.writer=H264VideoWriter(root/f"{name}.{container}",root/f"{name}_timestamps.csv",config,capture_fps,queue_size)
        self.sub=node.create_subscription(Image,config["source_topic"],self._callback,qos_profile_sensor_data)
    def start(self):
        self.writer.start();self.node.get_logger().info(f"{self.name}: encoder={self.writer.encoder}")
    def _callback(self,msg):
        receipt=time.monotonic_ns()
        try:
            if self.c["source_encoding"]=="bayer_grbg8":
                raw=self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
                bgr=cv2.cvtColor(np.asarray(raw),cv2.COLOR_BayerGRBG2BGR)
            else:bgr=self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
            stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
            if not self.writer.submit(bgr,stamp,receipt):
                self.node.get_logger().warning(f"{self.name}: encoder queue frame drop")
        except Exception as e:
            now=time.monotonic_ns()
            if now-self.warned>1_000_000_000:self.warned=now;self.node.get_logger().error(f"{self.name} frame rejected: {e}")
    def stop(self):return self.writer.stop()
