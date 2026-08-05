from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

def validate_transform(T, atol=1e-7):
    T=np.asarray(T,dtype=float)
    if T.shape != (4,4) or not np.isfinite(T).all(): raise ValueError("transform must be finite 4x4")
    if not np.allclose(T[3],[0,0,0,1],atol=atol): raise ValueError("invalid homogeneous row")
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=atol) or not np.isclose(np.linalg.det(R),1,atol=atol): raise ValueError("rotation is not proper orthonormal")

def ur_pose_to_matrix(pose6):
    p=np.asarray(pose6,dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all(): raise ValueError("pose6 must be finite shape (6,)")
    T=np.eye(4); T[:3,3]=p[:3]; T[:3,:3]=Rotation.from_rotvec(p[3:]).as_matrix(); return T

def matrix_to_ur_pose(T):
    validate_transform(T); T=np.asarray(T); return np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_rotvec()]

def quaternion_pose_to_matrix(position, quaternion_xyzw):
    p=np.asarray(position,float); q=np.asarray(quaternion_xyzw,float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or np.linalg.norm(q)==0: raise ValueError("invalid quaternion pose")
    T=np.eye(4); T[:3,3]=p; T[:3,:3]=Rotation.from_quat(q).as_matrix(); return T

def matrix_to_quaternion_pose(T):
    validate_transform(T); T=np.asarray(T); return T[:3,3].copy(),Rotation.from_matrix(T[:3,:3]).as_quat()

def relative_pose_action(current_T,next_T,frame="tool"):
    validate_transform(current_T); validate_transform(next_T)
    C=np.asarray(current_T); N=np.asarray(next_T)
    if frame=="tool":
        dp=C[:3,:3].T@(N[:3,3]-C[:3,3]); dr=Rotation.from_matrix(C[:3,:3].T@N[:3,:3]).as_rotvec()
    elif frame=="base":
        dp=N[:3,3]-C[:3,3]; dr=Rotation.from_matrix(N[:3,:3]@C[:3,:3].T).as_rotvec()
    else: raise ValueError("frame must be tool or base")
    return np.r_[dp,dr]

def apply_relative_pose_action(current_T,delta6,frame="tool"):
    validate_transform(current_T); C=np.asarray(current_T); d=np.asarray(delta6,float)
    if d.shape!=(6,) or not np.isfinite(d).all(): raise ValueError("delta6 must be finite")
    N=np.eye(4)
    if frame=="tool":
        N[:3,3]=C[:3,3]+C[:3,:3]@d[:3]; N[:3,:3]=C[:3,:3]@Rotation.from_rotvec(d[3:]).as_matrix()
    elif frame=="base":
        N[:3,3]=C[:3,3]+d[:3]; N[:3,:3]=Rotation.from_rotvec(d[3:]).as_matrix()@C[:3,:3]
    else: raise ValueError("frame must be tool or base")
    return N

def apply_base_euler_xyz_action(current_T,delta6):
    """Apply Bridge-style base-frame XYZ translation and Euler component delta."""
    validate_transform(current_T); C=np.asarray(current_T); d=np.asarray(delta6,float)
    if d.shape!=(6,) or not np.isfinite(d).all():raise ValueError("delta6 must be finite")
    current_euler=Rotation.from_matrix(C[:3,:3]).as_euler("xyz")
    N=np.eye(4);N[:3,3]=C[:3,3]+d[:3];N[:3,:3]=Rotation.from_euler("xyz",current_euler+d[3:]).as_matrix()
    return N
