import numpy as np
import pytest
from inference.safety import GripperDebouncer,adapt_gripper_semantics,limit_action,validate_target_position,gripper_transition
from inference.octo_ur5e_inference import select_frame_pair,select_synchronized_pair

def test_policy_action_limits_translation_rotation_and_gripper():
    value=limit_action([1,0,0,0,2,0,3],.003,.02)
    assert np.isclose(np.linalg.norm(value[:3]),.003)
    assert np.isclose(np.linalg.norm(value[3:6]),.02)
    assert value[6]==1

def test_open_high_gripper_is_adapted_to_executor_closed_high():
    np.testing.assert_allclose(adapt_gripper_semantics([1,2,3,4,5,6,.8],"open_high"),[1,2,3,4,5,6,.2])
    np.testing.assert_allclose(adapt_gripper_semantics([1,2,3,4,5,6,.8],"closed_high"),[1,2,3,4,5,6,.8])
    with pytest.raises(ValueError):adapt_gripper_semantics(np.zeros(7),"unknown")

def test_open_high_model_values_drive_executor_semantics():
    filt=GripperDebouncer(.7,.3,1,1)
    model_closed=adapt_gripper_semantics([0,0,0,0,0,0,0],"open_high")[6]
    model_open=adapt_gripper_semantics([0,0,0,0,0,0,1],"open_high")[6]
    assert model_closed==1 and filt.update(model_closed,0)==1
    assert model_open==0 and filt.update(model_open,1)==0

def test_policy_rejects_nonfinite_and_outside_workspace():
    with pytest.raises(ValueError):limit_action([np.nan]*7)
    with pytest.raises(ValueError):validate_target_position([0,0,0])

def test_gripper_hysteresis():
    assert gripper_transition(.5,0)==0
    assert gripper_transition(.8,0)==1
    assert gripper_transition(.5,1)==1
    assert gripper_transition(.2,1)==0
    assert gripper_transition(.69,0,.7,.3)==0
    assert gripper_transition(.31,1,.7,.3)==1

def test_gripper_debouncer_requires_consecutive_threshold_evidence():
    filt=GripperDebouncer(.9,.1,3,3);state=0
    for value in (.95,.2,.95,.96):state=filt.update(value,state)
    assert state==0
    state=filt.update(.97,state);assert state==1
    for value in (.05,.5,.05,.04):state=filt.update(value,state)
    assert state==1
    state=filt.update(.03,state);assert state==0

def test_select_frame_pair_uses_training_period_not_policy_period():
    history=[(0,"old"),(100_000_000,"wanted"),(150_000_000,"middle"),(200_000_000,"latest")]
    previous,current,gap=select_frame_pair(history,.1,.01)
    assert previous=="wanted" and current=="latest"
    assert gap==pytest.approx(.1)

def test_select_frame_pair_rejects_wrong_temporal_spacing():
    assert select_frame_pair([(0,"old"),(500_000_000,"latest")],.1,.06) is None

def test_select_synchronized_pair_matches_both_camera_timestamps():
    primary=[(100_000_000,"p0"),(200_000_000,"p1")]
    wrist=[(105_000_000,"w0"),(195_000_000,"w1")]
    p,w,gap,sync=select_synchronized_pair(primary,wrist,.1,.01,.01)
    assert p==["p0","p1"] and w==["w0","w1"]
    assert gap==pytest.approx(.1) and sync==pytest.approx(.005)

def test_select_synchronized_pair_rejects_camera_skew():
    primary=[(100_000_000,"p0"),(200_000_000,"p1")]
    wrist=[(1_000_000,"w0"),(300_000_000,"w1")]
    assert select_synchronized_pair(primary,wrist,.1,.01,.04) is None
