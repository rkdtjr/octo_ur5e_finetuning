import numpy as np

from octo_ur5e_collector.core.trajectory import metrics
from octo_ur5e_collector.core.trajectory_smoothing import smooth_joint_trajectory


def test_smoothing_preserves_time_critical_gripper_anchors():
    t=np.linspace(0,4,401)
    base=np.sin(t)[:,None]*np.ones((1,6))
    noise=.01*np.sin(np.arange(len(t))*2.1)[:,None]
    q=base+noise
    result,report=smooth_joint_trajectory(t,q,[2.0],.35,3,.20)
    event=int(np.argmin(np.abs(t-2.0)))
    assert np.array_equal(result[0],q[0])
    assert np.array_equal(result[event],q[event])
    assert np.array_equal(result[-1],q[-1])
    assert report["max_position_change_rad"]>0
    assert metrics(t,result)["max_acceleration_rad_s2"]<metrics(t,q)["max_acceleration_rad_s2"]
