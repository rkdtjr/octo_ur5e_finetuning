import json,numpy as np,pytest
from octo_ur5e_collector.core.gripper import GripperController
from octo_ur5e_collector.core.keyboard_commands import KeyboardCommandQueue
from octo_ur5e_collector.core.trajectory import *
from octo_ur5e_collector.core.replay_scheduler import ReplayScheduler
from octo_ur5e_collector.core.episode import create_episode,update_status
from octo_ur5e_collector.core.config import load_config
from octo_ur5e_collector.ros_adapters import preflight
from unittest.mock import patch
def gc(**kw):
    d={"semantic_open":0,"semantic_closed":1,"output_value_for_open":0.0,"output_value_for_closed":1.0,"command_on_change_only":True,"minimum_command_interval_sec":.2};d.update(kw);return d
def test_gripper():
    calls=[];c=GripperController(gc(),calls.append,clock=lambda:1)
    assert c.command_semantic(1,execute=False).success and calls==[]
    assert c.command_semantic(1,execute=False).reason=="unchanged"
    with pytest.raises(ValueError):c.output_from_semantic(3)
    assert GripperController(gc(output_value_for_open=1,output_value_for_closed=0)).output_from_semantic(0)==1
def test_keyboard():
    q=KeyboardCommandQueue({"open_keys":["o"],"close_keys":["c"],"finish_keys":["q"],"abort_keys":["esc"]})
    assert q.submit("O").semantic_state==0;assert q.get_nowait().kind=="gripper";assert q.submit("q").kind=="finish";assert q.submit("esc").kind=="abort"
def test_trajectory():
    assert np.allclose(reorder_joints(["b","a"],[2,1],["a","b"]),[1,2])
    with pytest.raises(ValueError):reorder_joints(["a","a"],[1,2],["a"])
    t=np.array([0.,1.,2.]);q=np.zeros((3,6));p=np.zeros((3,6));g=np.array([0,1,1])
    assert validate_arrays(t,q,g,p)["valid"] and not validate_arrays([0,0,1],q,g,p)["valid"]
    q[1,0]=np.nan;assert not validate_arrays(t,q,g,p)["valid"]
def test_scheduler():
    s=ReplayScheduler([{"time_sec":1},{"time_sec":2}],.5);assert len(s.advance(2.1))==2;assert s.feedback_stale(2,1);s.cancel();assert s.cancelled
def test_episode(tmp_path):
    c=load_config("collector/config/collector.yaml");c.data["storage"]["output_root"]=str(tmp_path)
    root=create_episode(c,"task","fixed");update_status(root,"aborted","test");assert json.loads((root/"status.json").read_text())["state"]=="aborted"
    with pytest.raises(FileExistsError):create_episode(c,"task","fixed")
def test_topic_once_ignores_ros_yaml_separator():
    result=type("Result",(),{"returncode":0,"stdout":"True\n---\n","stderr":""})()
    with patch("subprocess.run",return_value=result):
        assert preflight._topic_once("/program","data")==(True,"True")
