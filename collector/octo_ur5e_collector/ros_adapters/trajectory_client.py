from __future__ import annotations
import numpy as np

class TrajectoryClient:
    def __init__(self,node,action_name,joint_names,feedback_callback=None):
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        self.Action=FollowJointTrajectory; self.client=ActionClient(node,FollowJointTrajectory,action_name)
        self.joints=joint_names; self.feedback_callback=feedback_callback; self.goal_handle=None
    def make_goal(self,times,positions):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint
        goal=self.Action.Goal(); goal.trajectory.joint_names=list(self.joints)
        for t,q in zip(times,positions):
            p=JointTrajectoryPoint(); p.positions=np.asarray(q,float).tolist()
            sec=int(t); p.time_from_start=Duration(sec=sec,nanosec=int((float(t)-sec)*1e9)); goal.trajectory.points.append(p)
        return goal
    async def send(self,goal,timeout=5):
        if not self.client.wait_for_server(timeout_sec=timeout):raise RuntimeError("trajectory action unavailable")
        self.goal_handle=await self.client.send_goal_async(goal,feedback_callback=self.feedback_callback)
        if not self.goal_handle.accepted:raise RuntimeError("trajectory goal rejected")
        return await self.goal_handle.get_result_async()
    async def cancel(self):
        if self.goal_handle is not None:await self.goal_handle.cancel_goal_async()
