from __future__ import annotations
from pathlib import Path
import signal,subprocess

class RosbagRecorder:
    def __init__(self,output,topics,storage_id):
        self.output=Path(output); self.topics=list(topics); self.storage_id=storage_id; self.process=None; self.log=None
    def start(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.log=open(self.output.parent/"rosbag.log","w",encoding="utf-8")
        args=["ros2","bag","record","--storage",self.storage_id,"--output",str(self.output),*self.topics]
        self.process=subprocess.Popen(args,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        return self.process.pid
    def stop(self,timeout=10):
        if self.process is None:return None
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(2)
                except subprocess.TimeoutExpired:self.process.kill(); self.process.wait()
        code=self.process.returncode
        if self.log:self.log.close()
        return code
    def metadata_exists(self): return (self.output/"metadata.yaml").exists()
