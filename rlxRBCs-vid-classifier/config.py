
"""
config.py — Central configuration and lightweight CLI parser + CLI overrides

Purpose:
- Store and organize project-wide defaults (paths, training, models, transforms)
- Support command-line overrides for tuning experiments
- Expose a unified `cfg` namespace usable across modules

Usage:
    python train.py --epochs 15 --batch_size 8 --lr 1e-4
"""

import argparse
from pathlib import Path
import torch

# =============================
# Default Configuration 
# (dictionary structure)
# =============================

DEFAULTS = {
    # General, used by video_dataset.py and train.py
    "clip_duration": 5.0,
    "random_clip": True,
    "seed": 42, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",   #TODO correct if logic is duplicated

    # Paths, used by video_dataset.py and train.py
    "data_root": Path("data/videos"),
    "train_csv": Path("data/train_annotations.csv"),
    "val_csv": Path("data/val_annotations.csv"),
    "output_dir": Path("output/checkpoints"),  #TODO consider output/checkpoints
    "log_dir": Path("logs"),    #TODO log implementation

    # Model, used by video_transforms.py and model.py
    "num_classes": 4,
    "model_name": "rlxClassDetect_small",
    "input_frames": 8,
    "input_size": 112,
    "pretrained": False,
    "ptv_module": "slowfast",
    "model_depth": 50,

    # Training, used by train.py
    "batch_size": 4,
    "epochs": 10,
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "scheduler_tmax": 20,   # Scheduler

    # Augmentation, used by video_transforms.py   #TODO set these to appropriate values based on dataset actual Height Length sizes
    "train_transforms": {
        "short_side_min": 128,
        "short_side_max": 160,
        "crop_size": 112,
        "horizontal_flip_prob": 0.495,    # 0.5 is already a default value of the function
    },
    "val_transforms": {
        "short_side_min": 128,
        "crop_size": 112,
    },

}

# =============================
# Argument Parser
# =============================

def get_config():   #TODO be sure these aren't duplicated in train.py
    parser = argparse.ArgumentParser(description="Video classification configuration")

    # Core parameters you may override from CLI
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training/validation.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, help="Number of DataLoader workers.")
    parser.add_argument("--model_name", type=str, help="Model architecture name.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use.")
    parser.add_argument("--data_root", type=Path, help="Path to video dataset.")
    parser.add_argument("--output_dir", type=Path, help="Where to store checkpoints.")
    parser.add_argument("--log_dir", type=Path, help="Where to store logs.")
    parser.add_argument("--clip_duration", type=float, default=5.0, help="Max duration (in seconds) of sampled video clip.")
    parser.add_argument("--random_clip", action="store_true", help="Enable random start times for video clips.")
    parser.add_argument("--ptv_module", type=str, choices=["slowfast", "resnet"], help="Select between slowfast and resnet modules.")

    # Parse known args safely
    args, _ = parser.parse_known_args()

    # Merge CLI arguments into defaults
    config = DEFAULTS.copy()
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    # Ensure directories exist
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["log_dir"].mkdir(parents=True, exist_ok=True)

    # Convert to a lightweight namespace
    from types import SimpleNamespace
    return SimpleNamespace(**config)

# This allows easy use like:
# >>> from config import cfg
# >>> print(cfg.batch_size)
cfg = get_config()
