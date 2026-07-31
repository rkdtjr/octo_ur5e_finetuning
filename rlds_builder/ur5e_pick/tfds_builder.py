from __future__ import annotations
import numpy as np
import tensorflow_datasets as tfds
from .episodes import load_trajectory

class Ur5ePick(tfds.core.GeneratorBasedBuilder):
    VERSION=tfds.core.Version("1.0.0")
    def __init__(self,*args,plan=None,**kwargs):self._plan=plan;super().__init__(*args,**kwargs)
    def _info(self):
        step={"observation":tfds.features.FeaturesDict({"image_primary":tfds.features.Image(shape=(256,256,3)),"image_wrist":tfds.features.Image(shape=(256,256,3)),"wrist_valid":tfds.features.Scalar(np.bool_),"proprio":tfds.features.Tensor(shape=(14,),dtype=np.float32),"joint_position":tfds.features.Tensor(shape=(6,),dtype=np.float32),"tcp_pose":tfds.features.Tensor(shape=(7,),dtype=np.float32),"gripper_state":tfds.features.Scalar(np.int64)}),"action":tfds.features.Tensor(shape=(7,),dtype=np.float32),"language_instruction":tfds.features.Text(),"is_first":tfds.features.Scalar(np.bool_),"is_last":tfds.features.Scalar(np.bool_),"is_terminal":tfds.features.Scalar(np.bool_)}
        features=tfds.features.FeaturesDict({"steps":tfds.features.Dataset(step),"episode_metadata":tfds.features.FeaturesDict({"episode_id":tfds.features.Text(),"segment_id":tfds.features.Scalar(np.int64),"quality_grade":tfds.features.Text()})})
        return self.dataset_info_from_configs(features=features)
    def _split_generators(self,dl_manager):
        if self._plan is None:raise ValueError("dataset plan is required")
        splits={}
        for split in ("train","val"):
            entries=[x for x in self._plan["trajectories"] if x["split"]==split]
            if entries:splits[split]=self._generate_examples(entries)
        return splits
    def _generate_examples(self,entries):
        for entry in entries:yield load_trajectory(entry)
