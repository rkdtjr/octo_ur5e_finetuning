from __future__ import annotations
from dataclasses import dataclass
import time

@dataclass(frozen=True)
class DigitalOutputReadback:
    value:float; receipt_time_ns:int; receipt_ros_ns:int

@dataclass(frozen=True)
class DigitalOutputCommandStatus:
    service_success:bool; readback_confirmed:bool

class DigitalOutputGripperAdapter:
    """SetIO/IOStates adapter. Construct only after rclpy is initialized."""
    def __init__(self,node,config):
        from ur_msgs.msg import IOStates
        from ur_msgs.srv import SetIO
        self.node=node; self.c=config; self.SetIO=SetIO; self._readback=None
        self.last_command_status=DigitalOutputCommandStatus(False,False)
        self.client=node.create_client(SetIO,config["set_io_service"])
        self.sub=node.create_subscription(IOStates,config["io_states_topic"],self._io,10)
    def _io(self,msg):
        pin=self.c["output_pin"]
        for x in msg.digital_out_states:
            if x.pin==pin:self._readback=DigitalOutputReadback(float(x.state),time.monotonic_ns(),self.node.get_clock().now().nanoseconds)
    def latest_readback(self):return self._readback
    def send(self,value,timeout):
        import rclpy
        self.last_command_status=DigitalOutputCommandStatus(False,False)
        if not self.client.wait_for_service(timeout_sec=timeout):return False
        req=self.SetIO.Request(); req.fun=self.SetIO.Request.FUN_SET_DIGITAL_OUT; req.pin=self.c["output_pin"]; req.state=float(value)
        fut=self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node,fut,timeout_sec=timeout)
        if not fut.done() or fut.exception() is not None or not fut.result().success:return False
        if not self.c.get("readback_from_io_states",True):
            self.last_command_status=DigitalOutputCommandStatus(True,False);return True
        deadline=time.monotonic()+self.c.get("confirmation_timeout_sec",1.0)
        while time.monotonic()<deadline:
            rclpy.spin_once(self.node,timeout_sec=min(.05,deadline-time.monotonic()))
            if self._readback is not None and self._readback.value==float(value):
                self.last_command_status=DigitalOutputCommandStatus(True,True);return True
        self.last_command_status=DigitalOutputCommandStatus(True,False)
        return False
