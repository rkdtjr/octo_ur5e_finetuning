from __future__ import annotations
from pathlib import Path
import signal,subprocess,shutil

class RosbagRecorder:
    def __init__(self,output,topics,storage_id,storage_preset_profile="none"):
        self.output=Path(output); self.topics=list(topics); self.storage_id=storage_id
        self.storage_preset_profile=storage_preset_profile; self.process=None; self.log=None
    def start(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.log=open(self.output.parent/"rosbag.log","w",encoding="utf-8")
        args=[
            "ros2","bag","record",
            "--storage",self.storage_id,
            "--storage-preset-profile",self.storage_preset_profile,
            "--output",str(self.output),
            "--disable-keyboard-controls",
            "--topics",*self.topics,
        ]
        # ros2 bag has no reason to consume the operator terminal. Leaving stdin
        # inherited races with the collector's nonblocking keyboard reader.
        self.process=subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
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
    def export_single_mcap(self,target):
        files=list(self.output.glob("*.mcap"))
        if len(files)!=1:raise RuntimeError(f"expected one MCAP file in {self.output}, found {len(files)}")
        target=Path(target)
        if target.exists():raise FileExistsError(target)
        shutil.move(str(files[0]),target)
        metadata=self.output/"metadata.yaml"
        if metadata.exists():shutil.move(str(metadata),target.with_name("robot_states_metadata.yaml"))
        try:self.output.rmdir()
        except OSError:pass
        return target
