from __future__ import annotations

import numpy as np


def joint_tracking_metrics(errors, joint_names):
    values = np.asarray(errors, dtype=float)
    empty = {
        "joint_tracking_sample_count": 0,
        "joint_tracking_rmse": None,
        "joint_tracking_max_error": None,
        "joint_tracking_per_joint_rmse": None,
        "joint_tracking_per_joint_max_error": None,
    }
    if values.size == 0:
        return empty
    if values.ndim != 2 or values.shape[1] != len(joint_names):
        raise ValueError("joint tracking errors must be samples x joints")
    return {
        "joint_tracking_sample_count": int(values.shape[0]),
        "joint_tracking_rmse": float(np.sqrt(np.mean(values**2))),
        "joint_tracking_max_error": float(np.max(np.abs(values))),
        "joint_tracking_per_joint_rmse": {
            name: float(value)
            for name, value in zip(joint_names, np.sqrt(np.mean(values**2, axis=0)))
        },
        "joint_tracking_per_joint_max_error": {
            name: float(value)
            for name, value in zip(joint_names, np.max(np.abs(values), axis=0))
        },
    }


def age_metrics(values_ms, prefix):
    values = np.asarray(values_ms, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            f"{prefix}_median_ms": None,
            f"{prefix}_p95_ms": None,
            f"{prefix}_max_ms": None,
        }
    return {
        f"{prefix}_median_ms": float(np.median(values)),
        f"{prefix}_p95_ms": float(np.percentile(values, 95)),
        f"{prefix}_max_ms": float(np.max(values)),
    }


def duration_metrics(planned, execution, recording, total_wall):
    return {
        "trajectory_planned_duration_sec": float(planned),
        "trajectory_execution_duration_sec": None if execution is None else float(execution),
        "recording_duration_sec": float(recording),
        "total_command_wall_time_sec": float(total_wall),
    }


def gripper_event_record(scheduled, actual, semantic_state, output_value, output_pin,
                         service_success, readback_confirmed):
    return {
        "event": "gripper_transition",
        "scheduled_trajectory_time_sec": float(scheduled),
        "actual_command_time_sec": float(actual),
        "timing_error_sec": float(actual - scheduled),
        "semantic_state": int(semantic_state),
        "physical_output_value": float(output_value),
        "output_pin": int(output_pin),
        "set_io_service_success": bool(service_success),
        "io_readback_confirmed": bool(readback_confirmed),
    }


def evaluate_quality(report, execution, max_tcp_age_ms=100.0):
    """Return a compact GOOD/WARNING/BAD assessment with explainable checks."""
    checks=[]

    def upper(name, value, good_max, warning_max, unit=""):
        if value is None:
            rating="BAD"
        elif value <= good_max:
            rating="GOOD"
        elif value <= warning_max:
            rating="WARNING"
        else:
            rating="BAD"
        checks.append({"name":name,"rating":rating,"value":value,"unit":unit,
                       "good_max":good_max,"warning_max":warning_max})

    for camera in ("primary","wrist"):
        upper(f"{camera}_frame_drop_ratio",report.get(f"{camera}_frame_drop_ratio"),.01,.05,"ratio")
    upper("primary_wrist_time_difference_p95",report.get("primary_wrist_time_difference_p95_ms"),20.0,40.0,"ms")
    upper("pose_sync_p95",report.get("pose_sync_p95_ms"),20.0,40.0,"ms")
    upper("joint_sync_p95",report.get("joint_sync_p95_ms"),20.0,40.0,"ms")
    upper("gripper_age_p95",report.get("gripper_age_p95_ms"),50.0,100.0,"ms")
    upper("tcp_age_p95",report.get("tcp_age_p95_ms"),max_tcp_age_ms/2,max_tcp_age_ms,"ms")
    upper("joint_tracking_rmse",execution.get("joint_tracking_rmse"),.01,.03,"rad")
    upper("joint_tracking_max_error",execution.get("joint_tracking_max_error"),.05,.10,"rad")

    events=execution.get("gripper_events",[])
    if events:
        service_ok=all(x.get("set_io_service_success") for x in events)
        readback_ok=all(x.get("io_readback_confirmed") for x in events)
        max_timing=max(abs(float(x["timing_error_sec"])) for x in events)
        checks.append({"name":"gripper_set_io_and_readback","rating":"GOOD" if service_ok and readback_ok else "BAD",
                       "value":{"event_count":len(events),"all_set_io_success":service_ok,"all_readback_confirmed":readback_ok}})
        upper("gripper_timing_error_max",max_timing,.10,.25,"s")

    if execution.get("dataset_compatible") is False:
        checks.append({"name":"dataset_compatible","rating":"BAD","value":False})
    rank={"GOOD":0,"WARNING":1,"BAD":2}
    overall=max((x["rating"] for x in checks),key=rank.get,default="BAD")
    counts={rating:sum(x["rating"]==rating for x in checks) for rating in rank}
    problems=[x["name"] for x in checks if x["rating"]!="GOOD"]
    verdict={"GOOD":"ready","WARNING":"review_recommended","BAD":"not_ready"}[overall]
    return {"overall":overall,"verdict":verdict,"counts":counts,"problems":problems,"checks":checks}
