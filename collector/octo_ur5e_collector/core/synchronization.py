from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

def target_timestamps(start_ns:int,end_ns:int,rate_hz:float)->np.ndarray:
    if rate_hz<=0 or end_ns<start_ns:raise ValueError("invalid timeline")
    step=int(round(1e9/rate_hz));return np.arange(start_ns,end_ns+1,step,dtype=np.int64)

def select_nearest_unique(source_ns,target_ns,max_error_ns=None):
    source=np.asarray(source_ns,dtype=np.int64);targets=np.asarray(target_ns,dtype=np.int64)
    selected=[];errors=[];last=-1
    for target in targets:
        pos=int(np.searchsorted(source,target));candidates=[i for i in (pos-1,pos) if last<i<len(source)]
        if not candidates:selected.append(-1);errors.append(None);continue
        i=min(candidates,key=lambda x:abs(int(source[x])-int(target)));error=int(source[i])-int(target)
        if max_error_ns is not None and abs(error)>max_error_ns:selected.append(-1);errors.append(error)
        else:selected.append(i);errors.append(error);last=i
    return np.asarray(selected,int),np.asarray(errors,object)

def match_cameras(primary_ns,wrist_ns,rate_hz,max_error_ns):
    start=max(int(primary_ns[0]),int(wrist_ns[0]));end=min(int(primary_ns[-1]),int(wrist_ns[-1]))
    targets=target_timestamps(start,end,rate_hz)
    pi,pe=select_nearest_unique(primary_ns,targets,max_error_ns);wi,we=select_nearest_unique(wrist_ns,targets,max_error_ns)
    valid=(pi>=0)&(wi>=0)
    return {"target_ns":targets,"primary_index":pi,"wrist_index":wi,"primary_error_ns":pe,"wrist_error_ns":we,"valid":valid}

def interpolation_bracket(times_ns,target_ns):
    t=np.asarray(times_ns,dtype=np.int64);right=int(np.searchsorted(t,target_ns))
    if right==0 or right==len(t):raise ValueError("target outside interpolation range")
    left=right-1;alpha=(target_ns-t[left])/(t[right]-t[left]);return left,right,float(alpha)

def interpolate_linear(times_ns,values,target_ns):
    a,b,u=interpolation_bracket(times_ns,target_ns);v=np.asarray(values,float)
    return (1-u)*v[a]+u*v[b],int(times_ns[b]-times_ns[a])

def interpolate_quaternion(times_ns,quaternions_xyzw,target_ns):
    a,b,u=interpolation_bracket(times_ns,target_ns)
    rotations=Rotation.from_quat(np.asarray(quaternions_xyzw,float)[[a,b]])
    value=Slerp([0.0,1.0],rotations)([u]).as_quat()[0]
    return value,int(times_ns[b]-times_ns[a])

def latest_discrete(times_ns,values,target_ns):
    i=int(np.searchsorted(times_ns,target_ns,side="right"))-1
    if i<0:raise ValueError("no prior discrete value")
    return values[i],int(target_ns-times_ns[i])
