"""Selected-frame H.264 decode helpers for the future RLDS builder."""
from __future__ import annotations
import cv2
import numpy as np

def decode_selected_rgb(video_path,indices):
    wanted=sorted(set(int(x) for x in indices if int(x)>=0));out={}
    cap=cv2.VideoCapture(str(video_path));i=0;target_pos=0
    try:
        while target_pos<len(wanted):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f"video ended before selected frame {wanted[target_pos]}")
            if i==wanted[target_pos]:
                out[i]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                target_pos+=1
            i+=1
    finally:cap.release()
    return [out[int(i)] for i in indices if int(i)>=0]
