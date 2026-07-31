from dataclasses import dataclass
import time

@dataclass(frozen=True)
class GripperCommandResult:
    semantic_state:int; output_value:float; execute:bool; service_called:bool; success:bool; reason:str=""

class GripperController:
    def __init__(self,config,send_output=None,clock=time.monotonic):
        self.c=config; self.send=send_output; self.clock=clock; self.last_state=None; self.last_time=float("-inf")
    def output_from_semantic(self,state):
        if state not in (0,1): raise ValueError("semantic gripper state must be 0 or 1")
        key="output_value_for_open" if state==self.c["semantic_open"] else "output_value_for_closed"
        return float(self.c[key])
    def semantic_from_output(self,value):
        if value==self.c["output_value_for_open"]: return self.c["semantic_open"]
        if value==self.c["output_value_for_closed"]: return self.c["semantic_closed"]
        raise ValueError("output does not match configured polarity")
    def command_semantic(self,state,*,execute):
        value=self.output_from_semantic(state); now=self.clock()
        if self.c["command_on_change_only"] and state==self.last_state: return GripperCommandResult(state,value,execute,False,True,"unchanged")
        if now-self.last_time < self.c["minimum_command_interval_sec"]: return GripperCommandResult(state,value,execute,False,False,"minimum_interval")
        if not execute: self.last_state=state; self.last_time=now; return GripperCommandResult(state,value,False,False,True,"dry_run")
        if self.send is None: return GripperCommandResult(state,value,True,False,False,"no_adapter")
        ok=bool(self.send(value)); self.last_time=now
        if ok: self.last_state=state
        return GripperCommandResult(state,value,True,True,ok,"" if ok else "service_failed")
