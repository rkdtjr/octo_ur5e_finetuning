import numpy as np,pytest
from scipy.spatial.transform import Rotation
from octo_ur5e_collector.core.transforms import *
@pytest.mark.parametrize("pose",[[0,0,0,0,0,0],[1,2,3,0,0,0],[0,0,0,.2,-.3,.4],[0,0,0,1e-12,0,0],[0,0,0,np.pi-1e-8,0,0]])
def test_pose_round_trip(pose):
    T=ur_pose_to_matrix(pose);assert np.allclose(ur_pose_to_matrix(matrix_to_ur_pose(T)),T)
def test_quaternion_and_random():
    rng=np.random.default_rng(2)
    for _ in range(20):
        T=np.eye(4);T[:3,:3]=Rotation.random(random_state=rng).as_matrix();T[:3,3]=rng.normal(size=3)
        p,q=matrix_to_quaternion_pose(T);assert np.allclose(quaternion_pose_to_matrix(p,q),T)
@pytest.mark.parametrize("frame",["tool","base"])
def test_delta(frame):
    a=ur_pose_to_matrix([.2,-.1,.4,.3,.2,-.1]);b=ur_pose_to_matrix([.4,.3,.2,-.2,.5,.1])
    assert np.allclose(apply_relative_pose_action(a,relative_pose_action(a,b,frame),frame),b)
@pytest.mark.parametrize("T",[np.eye(3),np.diag([1,1,-1,1]),np.diag([1,2,1,1])])
def test_invalid(T):
    with pytest.raises(ValueError):validate_transform(T)

def test_bridge_base_euler_xyz_delta():
    current=np.eye(4);current[:3,3]=[.4,.2,.3];current[:3,:3]=Rotation.from_euler("xyz",[.1,-.2,.3]).as_matrix()
    delta=np.array([.01,-.02,.03,.04,.05,-.06])
    target=apply_base_euler_xyz_action(current,delta)
    np.testing.assert_allclose(target[:3,3],[.41,.18,.33])
    np.testing.assert_allclose(target[:3,:3],Rotation.from_euler("xyz",[.14,-.15,.24]).as_matrix())
