from __future__ import annotations

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from config import cfg
from datasets.video_dataset import CSVDataset
from transforms.video_transforms import train_transform, val_transform
from model import build_model
from evaluate import evaluate


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

    for videos, labels in tqdm(dataloader, desc="Training", leave=False):
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


# ---------------------------------------------------------
# Main training routine
# ---------------------------------------------------------
def main() -> None:
    """Entry point for training the rlxClassDetect model."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --------------------------
    # Dataset & DataLoaders
    # --------------------------
    train_set = CSVDataset(
        csv_path=cfg.train_csv,
        root_dir=cfg.data_root,
        transform=train_transform(),
    )

    val_set = CSVDataset(
        csv_path=cfg.val_csv,
        root_dir=cfg.data_root,
        transform=val_transform(),
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
    model = build_model(num_classes=cfg.num_classes).to(device)
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
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(cfg.output_dir, "best_model.pth"),
            )

    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
