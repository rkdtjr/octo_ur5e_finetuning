from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional

@dataclass(frozen=True)
class KeyboardCommand:
    kind: str
    key: str
    semantic_state: Optional[int]=None

class KeyboardCommandQueue:
    def __init__(self, config):
        self.queue=Queue(); self.mapping={}
        for k in config["open_keys"]: self.mapping[k]=("gripper",0)
        for k in config["close_keys"]: self.mapping[k]=("gripper",1)
        for k in config["finish_keys"]: self.mapping[k]=("finish",None)
        for k in config["abort_keys"]: self.mapping[k]=("abort",None)
    def submit(self,key):
        key=key.lower(); value=self.mapping.get(key)
        if value:
            cmd=KeyboardCommand(value[0],key,value[1]); self.queue.put(cmd); return cmd
    def get_nowait(self):
        try: return self.queue.get_nowait()
        except Empty: return None
