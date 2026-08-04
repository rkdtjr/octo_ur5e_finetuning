from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from .trajectory import metrics

def smooth_joint_trajectory(times,joints,gripper_event_times,window_sec=.35,polyorder=3,anchor_window_sec=.5):
    """Smooth joints without changing timestamps or critical gripper poses."""
    t=np.asarray(times,dtype=float);q=np.asarray(joints,dtype=float)
    if len(t)<5 or q.shape!=(len(t),6):raise ValueError("trajectory is too short for smoothing")
    dt=float(np.median(np.diff(t)));window=max(polyorder+2,int(round(window_sec/dt)))
    if window%2==0:window+=1
    window=min(window,len(t) if len(t)%2 else len(t)-1)
    if window<=polyorder:raise ValueError("smoothing window must exceed polyorder")
    filtered=savgol_filter(q,window_length=window,polyorder=polyorder,axis=0,mode="interp")
    anchors=np.asarray([t[0],*gripper_event_times,t[-1]],dtype=float)
    distance=np.min(np.abs(t[:,None]-anchors[None,:]),axis=1);blend=max(window_sec/2,dt)
    weight=np.clip((distance-anchor_window_sec)/blend,0.0,1.0);weight=weight*weight*(3-2*weight)
    result=q+(filtered-q)*weight[:,None];result[0]=q[0];result[-1]=q[-1]
    for event_time in gripper_event_times:result[int(np.argmin(np.abs(t-event_time)))]=q[int(np.argmin(np.abs(t-event_time)))]
    before=metrics(t,q);after=metrics(t,result)
    return result,{"enabled":True,"method":"savgol_with_gripper_anchors","window_sec":float(window_sec),"polyorder":int(polyorder),"anchor_window_sec":float(anchor_window_sec),"max_position_change_rad":float(np.max(np.abs(result-q))),"anchor_count":int(len(anchors)),"before":before,"after":after}
