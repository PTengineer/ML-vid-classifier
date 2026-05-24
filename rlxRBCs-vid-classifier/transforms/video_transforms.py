from ast import Tuple
import importlib, sys
# ModuleNotFoundError suggested shim
try:
    importlib.import_module("torchvision.transforms.functional_tensor")
except ModuleNotFoundError:
    try:
        mod = importlib.import_module("torchvision.transforms.functional")
        sys.modules["torchvision.transforms.functional_tensor"] = mod
    except ModuleNotFoundError:
        pass
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip

import torch

import torch.nn.functional as F
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    RandomResizedCrop,
    UniformTemporalSubsample,
)

from config import cfg


class PackPathway(torch.nn.Module):
    """
    Transform for packing video frames into SlowFast dual-pathway format.
    
    Converts a single [C, T, H, W] tensor into [slow_pathway, fast_pathway]
    where slow_pathway has temporal stride of alpha relative to fast_pathway.
    
    Args:
        alpha (int): Temporal stride for slow pathway. Default: 4.
    """
    def __init__(self, alpha: int = 4):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, frames: torch.Tensor) -> list:
        """
        Deterministically pack frames into SlowFast dual pathways using torch.index_select.
        
        Ensures exactly T // alpha frames in slow pathway, avoiding ceil() artifacts
        from stride-based slicing (e.g., [:, ::alpha, :, :]).
        
        Args:
            frames: Tensor of shape [C, T, H, W]
        
        Returns:
            [slow_pathway, fast_pathway]: List of two tensors with shapes
                slow: [C, T//alpha, H, W]
                fast: [C, T, H, W]
        """
        fast_pathway = frames
        num_slow_frames = frames.shape[1] // self.alpha
        slow_indices = torch.linspace(
            0, frames.shape[1] - 1, num_slow_frames
        ).long().to(frames.device)
        slow_pathway = torch.index_select(frames, 1, slow_indices)
        return [slow_pathway, fast_pathway]


# Top-level helpers (picklable) ------------------------------------------------
def to_float_div255(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.45, 0.45, 0.45], device=x.device).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], device=x.device).view(3, 1, 1, 1)
    return (x - mean) / std


def _resize_spatial(x: torch.Tensor, size: int) -> torch.Tensor:
    x_t = x.permute(1, 0, 2, 3)
    x_t_resized = F.interpolate(x_t, size=(size, size), mode="bilinear", align_corners=False)
    return x_t_resized.permute(1, 0, 2, 3)


def center_crop_to_input(x: torch.Tensor) -> torch.Tensor:
    """Center-crop or upscale video tensor spatially. Preserves C, T dims."""
    size = cfg.input_size
    
    # Validate expected shape [C, T, H, W]
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
    
    C, T, H, W = x.shape
    
    # Early exit: already target size
    if H == size and W == size:
        return x
    
    # Upscale if too small
    if H < size or W < size:
        return _resize_spatial(x, size)
    
    # Center crop (preserves C, T)
    top = (H - size) // 2
    left = (W - size) // 2
    return x[:, :, top : top + size, left : left + size]


def train_transform():
    """Video transform pipeline used during training."""
    train_cfg = cfg.train_transforms
    ptv_module = getattr(cfg, "ptv_module", "slowfast")  # default fallback


    if ptv_module.lower() == "slowfast":
        # For SlowFast, we apply spatial augmentations before packing pathways
        # to ensure both pathways receive the same spatial transformations.
        alpha = getattr(cfg, "slowfast_alpha", 4)
        return Compose([
            Lambda(center_crop_to_input),        
            UniformTemporalSubsample(cfg.input_frames),
            Lambda(to_float_div255),     
            Normalize(cfg.mean, cfg.std, inplace=False),
            RandomShortSideScale(min_size=train_cfg.short_side_min, max_size=train_cfg.short_side_max),
            RandomResizedCrop(
                target_height=train_cfg.crop_size,
                target_width=train_cfg.crop_size,
                scale=(0.7, 1.0),
                aspect_ratio=(0.75, 1.3333333333333333),
            ),
            RandomHorizontalFlip(),

            PackPathway(alpha=alpha),
        ])
    
    elif ptv_module.lower() == "resnet":
        # For ResNet3D, we can apply spatial augmentations directly without packing pathways.
        return Compose([
            Lambda(center_crop_to_input),        
            UniformTemporalSubsample(cfg.input_frames),
            Lambda(to_float_div255),     
            Normalize(cfg.mean, cfg.std, inplace=False),
            RandomShortSideScale(min_size=train_cfg.short_side_min, max_size=train_cfg.short_side_max),
            RandomResizedCrop(
                target_height=train_cfg.crop_size,
                target_width=train_cfg.crop_size,
                scale=(0.7, 1.0),
                aspect_ratio=(0.75, 1.3333333333333333),
            ),
            RandomHorizontalFlip(),
        ])


def val_transform():
    """Video transform pipeline for validation and inference (deterministic)."""
    ptv_module = getattr(cfg, "ptv_module", "slowfast")
    if ptv_module.lower() == "slowfast":    
        alpha = getattr(cfg, "slowfast_alpha", 4)

        return Compose([
            Lambda(center_crop_to_input),        
            UniformTemporalSubsample(cfg.input_frames),
            Lambda(to_float_div255),
            Normalize(cfg.mean, cfg.std, inplace=False),

            PackPathway(alpha=alpha),
        ])
    
    elif ptv_module.lower() == "resnet":
        return Compose([
            Lambda(center_crop_to_input),        
            UniformTemporalSubsample(cfg.input_frames),
            Lambda(to_float_div255),
            Normalize(cfg.mean, cfg.std, inplace=False),
        ])


def test_transform():
    """Video transform pipeline for overfit testing (minimal + deterministic)."""
    ptv_module = getattr(cfg, "ptv_module", "slowfast")
    if ptv_module.lower() == "slowfast":    
        alpha = getattr(cfg, "slowfast_alpha", 4)

        return Compose([
            Lambda(center_crop_to_input),        
            UniformTemporalSubsample(cfg.input_frames),
            Lambda(to_float_div255),
            Normalize(cfg.mean, cfg.std, inplace=False),

            PackPathway(alpha=alpha),
        ])
    
    elif ptv_module.lower() == "resnet":
        return Compose([
            Lambda(center_crop_to_input),        
            UniformTemporalSubsample(cfg.input_frames),
            Lambda(to_float_div255),
            Normalize(cfg.mean, cfg.std, inplace=False),
        ])    