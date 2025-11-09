from pytorchvideo.transforms import (
    RandomShortSideScale,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
import torch
from config import cfg

def train_transform():
    """
    Video transform pipeline used during training.

    Applies random spatial augmentations and normalization to improve
    model generalization. Temporal subsampling keeps clips lightweight.
    """

    train_cfg = cfg.train_transforms

    return Compose([
        # Uniformly sample n frames from the full clip
        UniformTemporalSubsample(cfg.input_frames),

        # Randomly resize the shorter video side between n and m pixels
        RandomShortSideScale(min_size=train_cfg.short_side_min, max_size=train_cfg.short_side_max),

        # Randomly crop a n x n spatial region
        RandomCropVideo(train_cfg.crop_size),

        # Randomly flip the clip horizontally (mirrors left↔right)
        RandomHorizontalFlipVideo(),

        # Scale raw pixel values from [0, 255] to [0, 1]
        Lambda(lambda x: x / 255.0),

        # Normalize each channel using ImageNet mean/std values
        Lambda(lambda x: (
            x - torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        ) / torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)),
    ])


def val_transform():
    """
    Video transform pipeline for validation and inference.

    Keeps deterministic preprocessing (no random crops or flips) so results
    are stable and directly comparable between epochs.
    """

    val_cfg = cfg.val_transforms

    return Compose([
        # Uniformly sample 8 frames from the full clip
        UniformTemporalSubsample(cfg.input_frames),

        # Normalize pixel intensities and channels as in training
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: (
            x - torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        ) / torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)),
    ])
