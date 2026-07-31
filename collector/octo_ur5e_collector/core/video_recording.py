from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import csv,json,queue,shutil,subprocess,threading,time
import numpy as np

@dataclass(frozen=True)
class FrameTimestamp:
    frame_index:int; ros_stamp_ns:int; monotonic_ns:int; receive_monotonic_ns:int

def available_ffmpeg_encoders()->set[str]:
    p=subprocess.run(["ffmpeg","-hide_banner","-encoders"],capture_output=True,text=True)
    return {x for x in ("h264_nvenc","libx264") if x in p.stdout}

def encoder_works(name:str)->bool:
    if shutil.which("ffmpeg") is None:return False
    args=["ffmpeg","-v","error","-f","lavfi","-i","color=size=64x64:rate=1","-frames:v","1","-c:v",name,"-f","null","-"]
    return subprocess.run(args,capture_output=True,timeout=10).returncode==0

def select_encoder(preferred:str,fallback:str)->str:
    for name in (preferred,fallback):
        if encoder_works(name):return name
    raise RuntimeError(f"neither encoder works: {preferred}, {fallback}")

class H264VideoWriter:
    """Bounded asynchronous raw-BGR to H.264 writer with accepted-frame CSV."""
    def __init__(self,path,csv_path,config,capture_fps,queue_size,encoder=None):
        self.path=Path(path);self.csv_path=Path(csv_path);self.c=config;self.fps=float(capture_fps)
        self.width,self.height=map(int,config["resolution"]);self.encoder=encoder or select_encoder(config["preferred_encoder"],config["fallback_encoder"])
        self.queue=queue.Queue(maxsize=int(queue_size));self.process=None;self.thread=None;self.error=None
        self.received=0;self.written=0;self.dropped=0;self.timestamps=[];self.started_ns=None;self.ended_ns=None
    def _args(self):
        preset=self.c["preset"]
        if preset=="auto":preset="p4" if self.encoder=="h264_nvenc" else "medium"
        return ["ffmpeg","-y","-loglevel","warning","-f","rawvideo","-pixel_format","bgr24","-video_size",f"{self.width}x{self.height}","-framerate",str(self.fps),"-i","pipe:0","-an","-c:v",self.encoder,"-preset",preset,"-profile:v",self.c["profile"],"-b:v",f"{self.c['bitrate_mbps']}M","-maxrate",f"{self.c['maxrate_mbps']}M","-bufsize",f"{self.c['bufsize_mbps']}M","-g",str(self.c["gop_size"]),"-pix_fmt",self.c["pixel_format"],str(self.path)]
    def start(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.process=subprocess.Popen(self._args(),stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        self.started_ns=time.monotonic_ns();self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def submit(self,bgr,ros_stamp_ns,receive_monotonic_ns)->bool:
        self.received+=1;frame=np.asarray(bgr)
        if frame.shape!=(self.height,self.width,3) or frame.dtype!=np.uint8:raise ValueError(f"invalid frame {frame.shape}/{frame.dtype}")
        item=(np.ascontiguousarray(frame).tobytes(),int(ros_stamp_ns),int(receive_monotonic_ns))
        try:self.queue.put_nowait(item);return True
        except queue.Full:self.dropped+=1;return False
    def _run(self):
        try:
            while True:
                item=self.queue.get()
                if item is None:break
                data,stamp,receipt=item
                self.process.stdin.write(data)
                self.timestamps.append(FrameTimestamp(self.written,stamp,receipt,receipt));self.written+=1
        except (BrokenPipeError,OSError) as e:self.error=str(e)
    def stop(self):
        self.queue.put(None);self.thread.join()
        if self.process.stdin:self.process.stdin.close()
        stderr=self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        code=self.process.wait(timeout=30);self.ended_ns=time.monotonic_ns()
        if code and self.error is None:self.error=stderr.strip() or f"ffmpeg exit {code}"
        with self.csv_path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FrameTimestamp.__dataclass_fields__);w.writeheader()
            for row in self.timestamps:w.writerow(asdict(row))
        return self.statistics()
    def statistics(self):
        stamps=np.array([x.ros_stamp_ns for x in self.timestamps],dtype=np.int64)
        intervals=np.diff(stamps)/1e6 if len(stamps)>1 else np.array([])
        duration=(stamps[-1]-stamps[0])/1e9 if len(stamps)>1 else 0.0
        size=self.path.stat().st_size if self.path.exists() else 0
        decoded=None
        if size:
            p=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","default=nw=1:nk=1",str(self.path)],capture_output=True,text=True)
            if p.returncode==0:
                try:decoded=int(p.stdout.strip())
                except ValueError:pass
        return {"encoder":self.encoder,"received_frame_count":self.received,"frame_count":self.written,"decoded_frame_count":decoded,"timestamp_row_count":len(self.timestamps),"queue_drop_count":self.dropped,"frame_drop_ratio":self.dropped/max(1,self.received),"actual_fps":(self.written-1)/duration if duration>0 else 0.0,"interval_mean_ms":float(intervals.mean()) if intervals.size else None,"interval_std_ms":float(intervals.std()) if intervals.size else None,"max_interval_ms":float(intervals.max()) if intervals.size else None,"file_size_bytes":size,"bitrate_actual_mbps":size*8/duration/1e6 if duration>0 else 0.0,"error":self.error}

def storage_projection(paths,duration_sec):
    total=sum(Path(p).stat().st_size for p in paths if Path(p).exists())
    rate=total/max(duration_sec,1e-9)
    return {"episode_duration_sec":duration_sec,"total_episode_size_bytes":total,"average_storage_rate_mib_per_sec":rate/1024**2,"expected_size_for_30_sec_gib":rate*30/1024**3,"expected_size_for_50_episodes_gib":total*50/1024**3,"expected_size_for_100_episodes_gib":total*100/1024**3}
