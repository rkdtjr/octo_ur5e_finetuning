import numpy as np

from octo_ur5e_collector.core.quality_metrics import age_metrics,duration_metrics,evaluate_quality,gripper_event_record,joint_tracking_metrics


def test_joint_tracking_metrics_are_global_and_per_joint():
    errors=np.array([[1.0,-2.0],[3.0,-4.0]])
    result=joint_tracking_metrics(errors,["a","b"])
    assert result["joint_tracking_sample_count"]==2
    assert np.isclose(result["joint_tracking_rmse"],np.sqrt(7.5))
    assert result["joint_tracking_max_error"]==4.0
    assert np.isclose(result["joint_tracking_per_joint_rmse"]["a"],np.sqrt(5.0))
    assert result["joint_tracking_per_joint_max_error"]=={"a":3.0,"b":4.0}


def test_quality_metrics_are_explicitly_null_only_without_samples():
    assert joint_tracking_metrics([],["a"])["joint_tracking_rmse"] is None
    result=age_metrics([1.0,2.0,10.0],"tcp_age")
    assert result["tcp_age_median_ms"]==2.0
    assert result["tcp_age_p95_ms"] is not None
    assert result["tcp_age_max_ms"]==10.0


def test_duration_and_gripper_fixture_are_populated():
    durations=duration_metrics(29.8,29.9,32.0,49.9)
    assert all(durations[key] is not None for key in durations)
    event=gripper_event_record(10.0,10.02,1,0.0,0,True,True)
    assert event["timing_error_sec"]!=0
    assert event["semantic_state"]==1
    assert event["physical_output_value"]==0.0
    assert event["set_io_service_success"] and event["io_readback_confirmed"]


def test_quality_evaluation_has_good_warning_bad_summary():
    report={
        "primary_frame_drop_ratio":0.0,"wrist_frame_drop_ratio":0.0,
        "primary_wrist_time_difference_p95_ms":25.0,
        "pose_sync_p95_ms":10.0,"joint_sync_p95_ms":10.0,
        "gripper_age_p95_ms":5.0,"tcp_age_p95_ms":20.0,
    }
    event=gripper_event_record(1.0,1.02,1,0.0,0,True,True)
    execution={"dataset_compatible":True,"joint_tracking_rmse":.002,
               "joint_tracking_max_error":.02,"gripper_events":[event]}
    result=evaluate_quality(report,execution,100.0)
    assert result["overall"]=="WARNING"
    assert result["verdict"]=="review_recommended"
    assert result["counts"]["GOOD"]>0
    assert result["counts"]["WARNING"]==1
    assert result["counts"]["BAD"]==0
