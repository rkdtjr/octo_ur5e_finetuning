"""UR5e configuration for the official Octo ``scripts/finetune.py``."""
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference


def get_config(config_string="head_only,language_conditioned"):
    mode, modality = config_string.split(",")
    if mode not in {"full", "head_only", "head_mlp_only"}:
        raise ValueError(f"unsupported finetuning mode: {mode}")
    if modality not in {"language_conditioned", "multimodal"}:
        raise ValueError(f"unsupported modality: {modality}")
    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    else:
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    steps = FieldReference(5_000)
    window = FieldReference(2)
    return ConfigDict({
        "pretrained_path": "hf://rail-berkeley/octo-small-1.5",
        "pretrained_step": None,
        "batch_size": 4,
        "shuffle_buffer_size": 2_048,
        "num_steps": steps,
        "log_interval": 50,
        "eval_interval": 500,
        "save_interval": 500,
        "save_dir": "./runs",
        "seed": 42,
        "wandb": {"project": "octo_ur5e", "group": "pick", "entity": None},
        "dataset_kwargs": {
            "name": "ur5e_pick",
            "data_dir": "./data/rlds/ur5e_pick_3val",
            "image_obs_keys": {"primary": "image_primary", "wrist": None},
            "proprio_obs_key": None,
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "action_normalization_mask": [True, True, True, True, True, True, False],
            "standardize_fn": None,
            "num_parallel_reads": 4,
            "num_parallel_calls": 4,
        },
        "modality": modality,
        "finetuning_mode": mode,
        "window_size": window,
        "optimizer": {
            "learning_rate": {"name": "cosine", "init_value": 0.0, "peak_value": 3e-4,
                              "warmup_steps": 500, "decay_steps": steps, "end_value": 0.0},
            "weight_decay": 0.01, "clip_gradient": 1.0,
            "frozen_keys": frozen_keys, "grad_accumulation_steps": None,
        },
        "val_kwargs": {"val_shuffle_buffer_size": 256, "num_val_batches": 8},
        "viz_kwargs": {"eval_batch_size": 4, "trajs_for_metrics": 8,
                       "trajs_for_viz": 0, "samples_per_state": 2},
        "traj_transform_kwargs": {
            "window_size": window, "action_horizon": 4,
            "goal_relabeling_strategy": None,
            "task_augment_strategy": "delete_task_conditioning",
            "task_augment_kwargs": {"keep_image_prob": 0.0},
            "num_parallel_calls": 4,
        },
        "frame_transform_kwargs": {
            "resize_size": {"primary": (256, 256), "wrist": (128, 128)},
            "image_augment_kwargs": {
                "primary": {"random_resized_crop": {"scale": [0.9, 1.0], "ratio": [0.95, 1.05]},
                            "random_brightness": [0.05], "random_contrast": [0.95, 1.05],
                            "augment_order": ["random_resized_crop", "random_brightness", "random_contrast"]},
            },
            "num_parallel_calls": 4,
        },
    })
