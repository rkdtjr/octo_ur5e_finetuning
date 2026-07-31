"""Preflight checks and command generation for Octo fine-tuning."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from training.check_dataset import inspect_dataset


def conda_python(conda_env: str) -> Path:
    executable=shutil.which("conda")
    if executable:
        candidate=Path(executable).resolve().parent.parent/"envs"/conda_env/"bin/python"
        if candidate.is_file():return candidate
    raise FileNotFoundError(f"cannot locate Python for Conda environment: {conda_env}")


def doctor(dataset_dir: Path, octo_repo: Path, conda_env: str) -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True,
    ) if shutil.which("nvidia-smi") else None
    try:octo_python=conda_python(conda_env)
    except FileNotFoundError:octo_python=None
    probe = subprocess.run(
        [str(octo_python), "-c", "import jax,flax,optax,octo; print(jax.devices())"],
        capture_output=True, text=True,
    ) if octo_python else None
    dataset = inspect_dataset(dataset_dir)
    checks = {
        "dataset": dataset["status"],
        "validation_episode_count": dataset["splits"]["val"]["episodes"],
        "train_validation_overlap": [],
        "octo_repo": (octo_repo / "scripts/finetune.py").is_file(),
        "conda_env": conda_env,
        "octo_environment": probe.stdout.strip() if probe and probe.returncode == 0 else None,
        "gpu": gpu.stdout.strip() if gpu and gpu.returncode == 0 else None,
    }
    checks["ready"] = bool(checks["gpu"] and checks["octo_repo"] and checks["octo_environment"])
    return checks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="octo-train")
    parser.add_argument("--dataset-dir", default="data/rlds/ur5e_pick_3val/ur5e_pick/1.0.0")
    parser.add_argument("--octo-repo", default=os.environ.get("OCTO_REPO", "/home/sixr/Desktop/octo"))
    parser.add_argument("--conda-env", default=os.environ.get("OCTO_CONDA_ENV", "octo_env"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    dataset_dir, octo_repo = Path(args.dataset_dir), Path(args.octo_repo)
    status = doctor(dataset_dir, octo_repo, args.conda_env)
    config = Path(__file__).with_name("finetune_config.py").resolve()
    data_root = dataset_dir.parent.parent.resolve()
    command = [
        str(conda_python(args.conda_env)), str((octo_repo / "scripts/finetune.py").resolve()),
        f"--config={config}:head_only,language_conditioned",
        "--name=ur5e_pick",
        f"--config.dataset_kwargs.data_dir={data_root}",
        f"--config.save_dir={Path('runs').resolve()}",
    ]
    if args.smoke:
        command += ["--debug", "--config.num_steps=10", "--config.eval_interval=10",
                    "--config.save_interval=10", "--config.optimizer.learning_rate.warmup_steps=2", "--config.batch_size=4",
                    "--config.viz_kwargs.trajs_for_viz=0",
                    "--config.viz_kwargs.eval_batch_size=4"]
    status["command"] = command
    if not args.execute:
        print(json.dumps(status, indent=2)); return
    if not status["ready"]:
        print(json.dumps(status, indent=2))
        raise SystemExit("training preflight failed")
    environment=os.environ.copy()
    environment.update({
        "PYTHONPATH":str(Path.cwd()), "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
        "WANDB_MODE":"disabled" if args.smoke else "offline",
        "WANDB_DISABLED":"true" if args.smoke else "false",
    })
    wandb_dir=Path("runs/wandb").resolve();wandb_dir.mkdir(parents=True,exist_ok=True)
    environment["WANDB_DIR"]=str(wandb_dir)
    raise SystemExit(subprocess.run(command,env=environment).returncode)


if __name__ == "__main__":
    main()
