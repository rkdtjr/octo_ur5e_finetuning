import csv,subprocess
from unittest.mock import patch
import numpy as np
from octo_ur5e_collector.core.video_recording import H264VideoWriter,select_encoder
from octo_ur5e_collector.wrist_bayer_camera_node import debayer_grbg

def config():
    return {"resolution":[32,24],"preferred_encoder":"h264_nvenc","fallback_encoder":"libx264","bitrate_mbps":1,"maxrate_mbps":1,"bufsize_mbps":2,"gop_size":10,"preset":"auto","profile":"high","pixel_format":"yuv420p"}
def test_encoder_fallback():
    with patch("octo_ur5e_collector.core.video_recording.encoder_works",side_effect=lambda x:x=="libx264"):
        assert select_encoder("h264_nvenc","libx264")=="libx264"
def test_all_frames_and_csv_match(tmp_path):
    video=tmp_path/"x.mkv";index=tmp_path/"x.csv";w=H264VideoWriter(video,index,config(),30,20,encoder="libx264");w.start()
    for i in range(10):assert w.submit(np.full((24,32,3),i,np.uint8),i*33_333_333,i)
    stats=w.stop();rows=list(csv.DictReader(index.open()))
    assert stats["frame_count"]==10 and stats["queue_drop_count"]==0 and len(rows)==10
    probe=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","default=nw=1:nk=1",video],capture_output=True,text=True)
    assert int(probe.stdout)==10
def test_grbg_conversion_known_shape():
    raw=np.array([[255,128,255,128],[128,0,128,0],[255,128,255,128],[128,0,128,0]],np.uint8)
    out=debayer_grbg(raw,4,4);assert out.shape==(4,4,3) and out.dtype==np.uint8
