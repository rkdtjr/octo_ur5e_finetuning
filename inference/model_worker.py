"""Persistent Python 3.10 Octo worker using newline-delimited JSON."""
from __future__ import annotations
import argparse,base64,json,sys
import cv2,jax,numpy as np
from octo.model.octo_model import OctoModel


def decode_rgb(value):
    array=np.frombuffer(base64.b64decode(value),np.uint8);bgr=cv2.imdecode(array,cv2.IMREAD_COLOR)
    if bgr is None:raise ValueError("invalid JPEG frame")
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",required=True);parser.add_argument("--step",type=int);parser.add_argument("--dataset-statistics-key");parser.add_argument("--samples",type=int,default=4);parser.add_argument("--use-wrist",action="store_true")
    args=parser.parse_args();load_kwargs={"step":args.step} if args.step is not None else {};model=OctoModel.load_pretrained(args.checkpoint,**load_kwargs)
    if args.dataset_statistics_key:
        if args.dataset_statistics_key not in model.dataset_statistics:raise KeyError(f"dataset statistics not found: {args.dataset_statistics_key}")
        statistics=model.dataset_statistics[args.dataset_statistics_key]["action"]
    else:statistics=model.dataset_statistics["action"]
    rng=jax.random.PRNGKey(42);tasks={}
    print(json.dumps({"ready":True,"step":args.step,"dataset_statistics_key":args.dataset_statistics_key,"use_wrist":args.use_wrist,"spec":model.get_pretty_spec()}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line);frames=np.stack([decode_rgb(x) for x in request["primary_jpeg_b64"]])
            if frames.shape!=(2,256,256,3):raise ValueError(f"expected two 256x256 frames, got {frames.shape}")
            instruction=request["instruction"]
            if instruction not in tasks:tasks[instruction]=model.create_tasks(texts=[instruction])
            rng,key=jax.random.split(rng);observation={"image_primary":frames[None],"timestep_pad_mask":np.ones((1,2),bool)}
            if args.use_wrist:
                wrist=np.stack([decode_rgb(x) for x in request["wrist_jpeg_b64"]])
                if wrist.shape!=(2,128,128,3):raise ValueError(f"expected two 128x128 wrist frames, got {wrist.shape}")
                observation["image_wrist"]=wrist[None]
            actions=model.sample_actions(observation,tasks[instruction],unnormalization_statistics=statistics,sample_shape=(args.samples,),rng=key)
            action_chunk=np.median(np.asarray(actions)[:,0,:,:],axis=0)
            print(json.dumps({"ok":True,"action_chunk":action_chunk.tolist()}),flush=True)
        except Exception as error:print(json.dumps({"ok":False,"error":f"{type(error).__name__}: {error}"}),flush=True)


if __name__=="__main__":main()
