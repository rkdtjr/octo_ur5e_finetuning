import numpy as np
from octo_ur5e_collector.core.video_recording import H264VideoWriter
from processing.convert_to_rlds import decode_selected_rgb
def config():
    return {"resolution":[32,24],"preferred_encoder":"libx264","fallback_encoder":"libx264","bitrate_mbps":1,"maxrate_mbps":1,"bufsize_mbps":2,"gop_size":10,"preset":"auto","profile":"high","pixel_format":"yuv420p"}

def test_decode_returns_only_selected_rgb(tmp_path):
    path=tmp_path/"x.mkv";w=H264VideoWriter(path,tmp_path/"x.csv",config(),30,20,encoder="libx264");w.start()
    for i in range(6):
        frame=np.zeros((24,32,3),np.uint8);frame[:,:,2]=i*30
        w.submit(frame,i*33_333_333,i)
    w.stop();selected=decode_selected_rgb(path,[1,4])
    assert len(selected)==2 and all(x.shape==(24,32,3) and x.dtype==np.uint8 for x in selected)
