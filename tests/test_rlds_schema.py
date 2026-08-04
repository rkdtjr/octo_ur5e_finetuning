import json
import numpy as np
import yaml

from rlds_builder.pipeline import doctor
from rlds_builder.ur5e_pick.episodes import center_crop_resize,plan_dataset
from processing.synchronize_episode import source_fingerprint


def test_center_crop_resize_contract():
    image=np.zeros((80,120,3),np.uint8);image[:,20:100,0]=255
    result=center_crop_resize(image,[32,32])
    assert result.shape==(32,32,3) and result.dtype==np.uint8


def test_tfds_builder_schema_can_be_initialized(tmp_path):
    tfds = __import__("tensorflow_datasets")
    from rlds_builder.ur5e_pick.tfds_builder import Ur5ePick

    builder = Ur5ePick(data_dir=str(tmp_path), plan={"trajectories": []})
    step = builder.info.features["steps"].feature
    assert step["observation"]["wrist_valid"].dtype == np.bool_
    assert step["observation"]["proprio"].shape == (14,)
    assert step["action"].shape == (7,)
    assert tfds.__version__


def test_dataset_plan_uses_valid_segments_and_quality(tmp_path):
    episode=tmp_path/"episode_a";(episode/"replay").mkdir(parents=True);(episode/"replay/robot_states.mcap").touch();(episode/"processed").mkdir()
    (episode/"processed/quality_report.json").write_text(json.dumps({"quality_grade":"GOOD"}))
    np.savez_compressed(episode/"processed/synchronized_episode.npz",segment_id=np.array([0,0,-1,1,1]),wrist_frame_indices=np.array([1,2,-1,4,5]),valid_mask=np.array([1,1,0,1,1],bool))
    (episode/"processed/processing_manifest.yaml").write_text(yaml.safe_dump({"source_fingerprint":source_fingerprint(episode)}))
    plan=plan_dataset(tmp_path)
    assert plan["trajectory_count"]==2
    assert [x["segment_id"] for x in plan["trajectories"]]==[0,1]
    assert plan["train_count"]+plan["val_count"]==2
    status=doctor(tmp_path)
    assert status["raw_episode_count"]==1 and status["current_processed_count"]==1


def test_dataset_plan_excludes_segments_without_a_transition(tmp_path):
    episode=tmp_path/"episode_a";(episode/"replay").mkdir(parents=True);(episode/"replay/robot_states.mcap").touch();(episode/"processed").mkdir()
    (episode/"processed/quality_report.json").write_text(json.dumps({"quality_grade":"GOOD"}))
    np.savez_compressed(episode/"processed/synchronized_episode.npz",segment_id=np.array([0,0,1]),wrist_frame_indices=np.array([1,2,3]),valid_mask=np.ones(3,bool))
    (episode/"processed/processing_manifest.yaml").write_text(yaml.safe_dump({"source_fingerprint":source_fingerprint(episode)}))
    plan=plan_dataset(tmp_path)
    assert [x["segment_id"] for x in plan["trajectories"]]==[0]
    assert any(x.get("segment_id")==1 and x["reason"]=="fewer_than_2_steps" for x in plan["skipped"])


def test_dataset_plan_keeps_whole_episodes_in_three_episode_validation_split(tmp_path):
    for number in range(5):
        episode=tmp_path/f"episode_{number}";(episode/"replay").mkdir(parents=True);(episode/"replay/robot_states.mcap").touch();(episode/"processed").mkdir()
        (episode/"processed/quality_report.json").write_text(json.dumps({"quality_grade":"GOOD"}))
        np.savez_compressed(episode/"processed/synchronized_episode.npz",segment_id=np.array([0,0,1,1]),wrist_frame_indices=np.arange(4),valid_mask=np.ones(4,bool))
        (episode/"processed/processing_manifest.yaml").write_text(yaml.safe_dump({"source_fingerprint":source_fingerprint(episode)}))
    plan=plan_dataset(tmp_path,val_episode_count=3)
    assert len(plan["val_episode_ids"])==3
    assert {x["episode_id"] for x in plan["trajectories"] if x["split"]=="val"}==set(plan["val_episode_ids"])
    assert not ({x["episode_id"] for x in plan["trajectories"] if x["split"]=="train"}&set(plan["val_episode_ids"]))
