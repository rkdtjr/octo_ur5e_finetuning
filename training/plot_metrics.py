"""Export local W&B offline history to CSV, JSON, and a loss plot."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys


def _find_wandb_file(value: Path) -> tuple[Path, Path]:
    if value.is_file() and value.suffix == ".wandb":
        return value, value.parent
    run_name=value.name
    candidates=[]
    for root in (Path("runs/wandb"),Path("/tmp/wandb")):
        if root.exists():candidates.extend(root.glob(f"**/*{run_name}*/*.wandb"))
    if not candidates:raise FileNotFoundError(f"offline W&B log not found for: {value}")
    return max(candidates,key=lambda path:path.stat().st_mtime),value


def _ensure_wandb_runtime(argv: list[str]) -> None:
    try:from wandb.sdk.internal.datastore import DataStore  # noqa: F401
    except ImportError:
        from training.doctor import conda_python
        python=conda_python(os.environ.get("OCTO_CONDA_ENV","octo_env"))
        os.execv(str(python),[str(python),str(Path(__file__).resolve()),*argv])


def read_history(path: Path) -> list[dict]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    store=DataStore();store.open_for_scan(str(path));rows=[]
    while True:
        data=store.scan_data()
        if data is None:break
        record=wandb_internal_pb2.Record();record.ParseFromString(data)
        if record.WhichOneof("record_type")!="history":continue
        row={item.key:json.loads(item.value_json) for item in record.history.item}
        rows.append(row)
    return rows


def export_metrics(run: Path,output: Path|None=None) -> dict:
    wandb_file,default_output=_find_wandb_file(run);rows=read_history(wandb_file)
    train=[{"step":int(x["_step"])+1,"loss":float(x["training/loss"])} for x in rows if "training/loss" in x]
    val=[{"step":int(x["_step"])+1,"loss":float(x["validation_ur5e_pick/base/loss"])} for x in rows if "validation_ur5e_pick/base/loss" in x]
    if not train or not val:raise ValueError("training or validation loss is missing from offline log")
    destination=output or default_output;destination.mkdir(parents=True,exist_ok=True)
    with (destination/"metrics.csv").open("w",newline="") as file:
        writer=csv.writer(file);writer.writerow(["split","step","loss"])
        writer.writerows(("train",x["step"],x["loss"]) for x in train)
        writer.writerows(("val",x["step"],x["loss"]) for x in val)
    import matplotlib.pyplot as plt
    import numpy as np
    steps=np.array([x["step"] for x in train]);loss=np.array([x["loss"] for x in train])
    window=min(10,len(loss));smooth=np.convolve(loss,np.ones(window)/window,mode="valid")
    figure,axis=plt.subplots(figsize=(9,5));axis.plot(steps,loss,alpha=.25,label="train loss")
    axis.plot(steps[window-1:],smooth,label=f"train moving avg ({window})")
    axis.plot([x["step"] for x in val],[x["loss"] for x in val],"o-",linewidth=2,label="validation loss")
    axis.set(xlabel="training step",ylabel="loss",title="Octo UR5e fine-tuning loss");axis.grid(alpha=.25);axis.legend();figure.tight_layout();figure.savefig(destination/"loss.png",dpi=160);plt.close(figure)
    summary={"status":"GOOD" if val[-1]["loss"]<=val[0]["loss"] else "WARNING","wandb_file":str(wandb_file),"train_points":len(train),"validation":val,"final_train_loss":train[-1]["loss"],"final_validation_loss":val[-1]["loss"],"outputs":{"csv":str(destination/"metrics.csv"),"plot":str(destination/"loss.png")}}
    (destination/"metrics_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main(argv=None) -> None:
    argv=list(sys.argv[1:] if argv is None else argv);_ensure_wandb_runtime(argv)
    parser=argparse.ArgumentParser(prog="octo-plot-metrics");parser.add_argument("run");parser.add_argument("--output")
    args=parser.parse_args(argv)
    try:print(json.dumps(export_metrics(Path(args.run),Path(args.output) if args.output else None),indent=2))
    except (FileNotFoundError,ValueError) as error:parser.error(str(error))


if __name__=="__main__":main()
