from __future__ import annotations

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from config import cfg
from datasets.video_dataset import RBCsDataset
from labels import load_label_mapping
from transforms.video_transforms import train_transform, val_transform
from models.model import build_model
from validate import evaluate 

import platform
import logger

import numpy as np
import random

# ---------------------------------------------------------
# Reproducability Seeding
# ---------------------------------------------------------
def set_seed(seed: int = 101) -> None:
    """Set random seed for reproducibility across numpy, torch, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one full epoch.

    Args:
        model: PyTorch model to train.
        dataloader: Iterable returning (videos, labels).
        criterion: Loss function such as CrossEntropyLoss.
        optimizer: Optimizer such as SGD or Adam.
        device: Target device ('cuda' or 'cpu').

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    accumulation_steps = cfg.acc_steps  # For future use if gradient accumulation is needed

    for i, (videos, labels) in tqdm(dataloader, desc="Training", leave=False):
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping (optional, can help with stability)
        if (i + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

# --------------------------
# Centralized logging
# --------------------------
LOG_NAME = f'{platform.node()}-rlxRBCrecognition'
LOG_FILE = 'train_log.txt'
log = logger.get_logger(LOG_NAME, LOG_FILE)

# --------------------------
# Configuration Validation
# --------------------------
def validate_config() -> None:
    """Validate configuration before training."""
    assert cfg.batch_size > 0, "batch_size must be positive"
    assert cfg.epochs > 0, "epochs must be positive"
    assert cfg.lr > 0, "learning rate must be positive"

    from pathlib import Path
    assert Path(cfg.data_root).exists(), f"data_root not found: {cfg.data_root}"
    assert Path(cfg.train_csv).exists(), f"train_csv not found: {cfg.train_csv}"
    assert Path(cfg.val_csv).exists(), f"val_csv not found: {cfg.val_csv}"

    if cfg.device == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA not available but device='cuda' specified"
        )

# ---------------------------------------------------------
# Main training routine
# ---------------------------------------------------------
def main() -> None:
    set_seed(cfg.seed)
    
    """Entry point for training the rlxClassDetect model."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Ensure model-specific directory exists for checkpoint saving
    os.makedirs(os.path.join(cfg.output_dir, cfg.model_name), exist_ok=True)

    validate_config()

    # --------------------------
    # Dataset & DataLoaders
    # --------------------------
    # Build authoritative class mapping from the training CSV and pass into datasets
    class_to_idx = load_label_mapping(cfg.train_csv)

    train_set = RBCsDataset(
        csv_file=cfg.train_csv,
        video_dir=cfg.data_root,
        transform=train_transform(),
        class_to_idx=class_to_idx,
    )

    val_set = RBCsDataset(
        csv_file=cfg.val_csv,
        video_dir=cfg.data_root,
        transform=val_transform(),
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # --------------------------
    # Model, loss, optimizer
    # --------------------------
    num_classes = len(class_to_idx) # for when a test dataset is used with 2 classes
    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.epochs,
    )

    best_val_acc = 0.0

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(cfg.epochs):
        try:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
        except RuntimeError as e:
            log.error(f"Epoch {epoch+1} failed: {e}. Resuming from best checkpoint.")
            model.load_state_dict(
                torch.load(os.path.join(cfg.output_dir, cfg.model_name, "_best_model.pth"))
            )
            continue

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()

        log.info(
            f"Epoch {epoch+1:3d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:>.4f} | Train Acc: {train_acc:>.4f} | "
            f"Val Loss: {val_loss:>.4f} | Val Acc: {val_acc:>.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_accuracy": best_val_acc,
                "cfg": cfg.__dict__, # Save config for reproducibility
            }
            
            torch.save(
                checkpoint,
                os.path.join(cfg.output_dir, cfg.model_name, "_best_model.pth"),
            )

    log.info(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training finished, details written to log file. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
