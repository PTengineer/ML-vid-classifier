from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model for a full epoch.

    Runs the model in evaluation mode, computes loss and top-1 accuracy
    across the entire validation/test dataset.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader providing (inputs, targets) batches.
        criterion: Loss function used for evaluation (e.g., CrossEntropyLoss).
        device: Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        (avg_loss, avg_accuracy):
            avg_loss (float): Mean loss over all batches.
            avg_accuracy (float): Mean accuracy in [0, 1] range.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Inference-only mode disables gradient storage for speed/memory
    with torch.inference_mode():
        for inputs, targets in dataloader:
            if isinstance(inputs, list):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Loss accumulation (sum, normalized later)
            loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            # Accuracy accumulation
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()

            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    # Normalize to per-sample averages
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    # Restore mode back to traning
    model.train()

    return avg_loss, avg_accuracy
