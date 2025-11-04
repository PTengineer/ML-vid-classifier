from pytorchvideo.transforms import (
    RandomShortSideScale,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
import torch

def train_transform():
    """
    Video transform pipeline used during training.

    Applies random spatial augmentations and normalization to improve
    model generalization. Temporal subsampling keeps clips lightweight.
    """
    return Compose([
        # Uniformly sample 8 frames from the full clip
        UniformTemporalSubsample(8),

        # Randomly resize the shorter video side between 128 and 160 pixels
        RandomShortSideScale(min_size=128, max_size=160),

        # Randomly crop a 112×112 spatial region
        RandomCropVideo(112),

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
    return Compose([
        # Uniformly sample 8 frames from the full clip
        UniformTemporalSubsample(8),

        # Normalize pixel intensities and channels as in training
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: (
            x - torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        ) / torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)),
    ])
