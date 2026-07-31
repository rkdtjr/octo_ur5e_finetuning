import numpy as np,pytest
from scipy.spatial.transform import Rotation
from octo_ur5e_collector.core.synchronization import *
from processing.synchronize_episode import robot_match
def test_timestamp_10hz_not_modulo_and_unique():
    source=np.array([0,31,68,101,139,201,234,269,303])*1_000_000
    targets=target_timestamps(0,300_000_000,10)
    idx,err=select_nearest_unique(source,targets,45_000_000)
    assert idx.tolist()==[0,3,5,8] and len(set(idx))==len(idx)
def test_common_camera_matching():
    p=np.arange(0,501,33,dtype=np.int64)*1_000_000;w=(np.arange(0,501,34,dtype=np.int64)+3)*1_000_000
    r=match_cameras(p,w,10,40_000_000);assert r["valid"].all();assert len(set(r["primary_index"]))==len(r["primary_index"])
def test_linear_slerp_and_discrete():
    t=np.array([0,100]);v=np.array([[0,0],[10,20]])
    x,gap=interpolate_linear(t,v,50);assert np.allclose(x,[5,10]) and gap==100
    q=Rotation.from_euler("z",[0,90],degrees=True).as_quat();mid,_=interpolate_quaternion(t,q,50)
    assert np.allclose(Rotation.from_quat(mid).as_euler("zxy",degrees=True)[0],45)
    value,age=latest_discrete(t,[0,1],75);assert value==0 and age==75
    with pytest.raises(ValueError):interpolate_linear(t,v,-1)
def test_robot_match_interpolation_and_gripper_zoh():
    states=[
        {"ros_stamp_ns":0,"actual_joint_positions":[0]*6,"actual_tcp":{"position":[0,0,0],"quaternion_xyzw":[0,0,0,1]},"gripper_state":0},
        {"ros_stamp_ns":100_000_000,"actual_joint_positions":[1]*6,"actual_tcp":{"position":[1,0,0],"quaternion_xyzw":Rotation.from_euler("z",90,degrees=True).as_quat().tolist()},"gripper_state":1},
    ]
    r=robot_match(states,50_000_000,{"max_pose_interpolation_gap_ms":110,"max_joint_interpolation_gap_ms":110,"max_gripper_age_ms":60})
    assert np.allclose(r["tcp_position"],[.5,0,0]) and np.allclose(r["joint_position"],[.5]*6)
    assert r["gripper_state"]==0 and r["robot_valid"]

def test_robot_match_uses_source_timestamps_and_rejects_stale_tcp():
    states=[
        {"ros_stamp_ns":10,"joint_source_ros_ns":0,"tcp_source_ros_ns":0,"tcp_valid":True,"gripper_source_ros_ns":0,"actual_joint_positions":[0]*6,"actual_tcp":{"position":[0,0,0],"quaternion_xyzw":[0,0,0,1]},"gripper_state":0},
        {"ros_stamp_ns":110_000_010,"joint_source_ros_ns":100_000_000,"tcp_source_ros_ns":100_000_000,"tcp_valid":True,"gripper_source_ros_ns":0,"actual_joint_positions":[1]*6,"actual_tcp":{"position":[1,0,0],"quaternion_xyzw":[0,0,0,1]},"gripper_state":0},
        {"ros_stamp_ns":120_000_010,"joint_source_ros_ns":110_000_000,"tcp_source_ros_ns":100_000_000,"tcp_valid":False,"gripper_source_ros_ns":0,"actual_joint_positions":[1]*6,"actual_tcp":None,"gripper_state":0},
    ]
    r=robot_match(states,50_000_000,{"max_pose_interpolation_gap_ms":110,"max_joint_interpolation_gap_ms":110,"max_gripper_age_ms":60,"max_tcp_age_ms":100})
    assert np.allclose(r["tcp_position"],[.5,0,0])
    assert r["pose_interpolation_gap_ms"]==100.0
