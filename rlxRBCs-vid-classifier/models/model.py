
"""
- Construct a 3D CNN (ResNet or SlowFast) for video classification
- Centralize all model hyperparameters and architecture decisions
- Keep compatibility with config.py for reproducible experiments

Usage:
    from model import build_model
    model = build_model(cfg)
"""

#import torch
import torch.nn as nn
import pytorchvideo.models.resnet as resnet
import pytorchvideo.models.slowfast as slowfast
from config import cfg

def build_model():
    """
    This constructs a video classification model and allows 
    a selection between modules.

    Supports both ResNet3D and SlowFast. We initially opt for SlowFast.

    Args:
            
    Returns:
            torch.nn.Module: Initialized model ready for training/inference  
    """

    model = None
    ptv_module = getattr(cfg, "ptv_module", "slowfast")  # default fallback


    if ptv_module.lower() == "slowfast":
        model = slowfast.create_slowfast(
            input_channels=3,
            model_depth=cfg.model_depth,
            model_num_class=cfg.num_classes,
            slowfast_channel_reduction_ratio=8,
            slowfast_conv_channel_ratio=2,
            dropout_rate=0.5,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            head_activation=None,
        )
    elif ptv_module.lower() == "resnet":
        model = resnet.create_resnet(
            input_channel=3,
            model_depth=cfg.model_depth,
            model_num_class=cfg.num_classes,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            stem_conv_kernel_size=(3, 7, 7),
            stem_conv_stride=(1, 2, 2),
            pool=nn.AvgPool3d,
        )
    else:
        raise ValueError(f"Unsupported ptv_module type: {ptv_module}")

    # Optionally load pretrained weights
    if getattr(cfg, "pretrained", False):
        print(f"[INFO] Loading pretrained weights for {ptv_module}...")
        # Placeholder — customize depending on where pretrained models are stored
        # model.load_state_dict(torch.load(cfg.pretrained_path))

    return model
