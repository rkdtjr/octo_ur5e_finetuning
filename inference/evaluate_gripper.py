"""Offline teacher-forced gripper evaluation on an RLDS split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from octo.model.octo_model import OctoModel


def _first_sustained(values, threshold, count, start, stop):
    start=max(0,start);stop=min(len(values),stop)
    for index in range(start,stop-count+1):
        if np.all(values[index:index+count]>=threshold):return index
    return None


def _resize_wrist(rgb):
    return cv2.resize(rgb,(128,128),interpolation=cv2.INTER_AREA)


def evaluate(args):
    tf.config.set_visible_devices([],"GPU")
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    model=OctoModel.load_pretrained(str(Path(args.checkpoint).resolve()),step=args.step)
    statistics=model.dataset_statistics["action"]
    builder=tfds.builder_from_directory(str(Path(args.dataset).resolve()))
    rng=jax.random.PRNGKey(args.seed);rows=[];timing=[]
    thresholds=np.arange(.5,.951,.05)
    for episode in builder.as_dataset(split=args.split):
        episode_id=episode["episode_metadata"]["episode_id"].numpy().decode()
        segment_id=int(episode["episode_metadata"]["segment_id"].numpy())
        steps=list(episode["steps"].as_numpy_iterator())
        if len(steps)<2:continue
        instruction=steps[0]["language_instruction"].decode()
        task=model.create_tasks(texts=[instruction]);predicted=[];ground_truth=[]
        for index in range(1,len(steps)):
            window=steps[index-1:index+1]
            primary=np.stack([x["observation"]["image_primary"] for x in window])
            wrist=np.stack([_resize_wrist(x["observation"]["image_wrist"]) for x in window])
            observation={"image_primary":primary[None],"image_wrist":wrist[None],"timestep_pad_mask":np.ones((1,2),bool)}
            rng,key=jax.random.split(rng)
            actions=model.sample_actions(observation,task,unnormalization_statistics=statistics,sample_shape=(args.samples,),rng=key)
            value=float(np.median(np.asarray(actions)[:,0,0,6]));target=float(steps[index]["action"][6])
            predicted.append(value);ground_truth.append(target)
            rows.append({"episode_id":episode_id,"segment_id":segment_id,"step":index,"time_sec":index/args.rate_hz,"instruction":instruction,"ground_truth_gripper":target,"predicted_gripper":value,"wrist_valid":bool(steps[index]["observation"]["wrist_valid"])})
        predicted=np.asarray(predicted);ground_truth=np.asarray(ground_truth)
        close_indices=np.flatnonzero((ground_truth[1:]>=.5)&(ground_truth[:-1]<.5))+1
        for close_index in close_indices:
            for threshold in thresholds:
                for debounce in (1,2,3):
                    crossing=_first_sustained(predicted,float(threshold),debounce,close_index-int(args.rate_hz),close_index+int(args.rate_hz)+1)
                    timing.append({"episode_id":episode_id,"segment_id":segment_id,"ground_truth_close_step":int(close_index+1),"threshold":round(float(threshold),2),"debounce_steps":debounce,"predicted_close_step":None if crossing is None else int(crossing+1),"timing_error_sec":None if crossing is None else float((crossing-close_index)/args.rate_hz)})
        figure,axis=plt.subplots(figsize=(10,3));times=np.arange(1,len(steps))/args.rate_hz
        axis.plot(times,ground_truth,label="ground truth",drawstyle="steps-post");axis.plot(times,predicted,label="model median")
        axis.axhline(.7,color="tab:red",linestyle="--",alpha=.5);axis.set_ylim(-.05,1.05);axis.set_xlabel("seconds");axis.set_ylabel("gripper");axis.legend();figure.tight_layout()
        figure.savefig(output/f"{episode_id}_segment_{segment_id:03d}.png",dpi=140);plt.close(figure)
    with (output/"predictions.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/"timing.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(timing[0]) if timing else ["episode_id"]);writer.writeheader();writer.writerows(timing)
    summary={"dataset":str(Path(args.dataset).resolve()),"checkpoint":str(Path(args.checkpoint).resolve()),"step":args.step,"split":args.split,"trajectory_count":len({(x['episode_id'],x['segment_id']) for x in rows}),"sample_count":len(rows),"ground_truth_close_count":len({(x['episode_id'],x['segment_id'],x['ground_truth_close_step']) for x in timing}),"thresholds":[]}
    for threshold in thresholds:
        for debounce in (1,2,3):
            selected=[x for x in timing if x["threshold"]==round(float(threshold),2) and x["debounce_steps"]==debounce];errors=[x["timing_error_sec"] for x in selected if x["timing_error_sec"] is not None]
            summary["thresholds"].append({"close_threshold":round(float(threshold),2),"debounce_steps":debounce,"detected":len(errors),"total":len(selected),"timing_error_median_sec":None if not errors else float(np.median(errors)),"timing_error_min_sec":None if not errors else float(np.min(errors)),"timing_error_max_sec":None if not errors else float(np.max(errors))})
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",required=True);parser.add_argument("--checkpoint",required=True);parser.add_argument("--step",type=int,default=10000);parser.add_argument("--split",default="val");parser.add_argument("--output",default="runs/gripper_eval");parser.add_argument("--samples",type=int,default=4);parser.add_argument("--seed",type=int,default=42);parser.add_argument("--rate-hz",type=float,default=10.)
    evaluate(parser.parse_args())


if __name__=="__main__":main()
