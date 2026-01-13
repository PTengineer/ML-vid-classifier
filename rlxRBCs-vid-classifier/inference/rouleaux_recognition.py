"""
rouleaux_recognition.py
-----------------------

Standalone inference script for detecting rouleaux and related ultrasound
artifacts from video files.

This script loads a trained PyTorchVideo model, applies the appropriate
validation transforms, performs forward inference, and returns a class
prediction with a confidence score.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Any, Union

import torch
from torch import nn

import platform

# --------------------------
# Project-Specific Imports
# --------------------------
from config import cfg
from model import build_model
from transforms.video_transforms import val_transform
import logger
from labels import load_class_labels    

# --------------------------
# Centralized logging
# --------------------------
LOG_NAME = f'{platform.node()}-rlxRBCrecognition'
LOG_FILE = 'inference_log.txt'
log = logger.get_logger(LOG_NAME, LOG_FILE)

# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def load_model_for_inference(
    weights_path: str,
    num_classes: int,
    device: torch.device
) -> nn.Module:
    """
    Loads and prepares a trained model for inference.

    Returns:
        A fully loaded PyTorch model in eval() mode.
    """
    log.info(f"Loading model weights from: {weights_path}")

    # Build model architecture using your modular builder
    model = build_model()

    # Load weights with device-safe mapping
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        log.error(f"Model weights file missing: {weights_path}")
        raise
    except RuntimeError as e:
        log.error(f"Error loading state_dict: {e}")
        log.error("Check that num_classes matches training settings.")
        raise

    model.to(device)
    model.eval()

    log.info("Model loaded successfully and set to evaluation mode.")
    return model


# ---------------------------------------------------------------------
# Inference on a Single Video
# ---------------------------------------------------------------------
def predict_video_artifact(
    model: nn.Module,
    video_path: str,
    device: torch.device,
    transform_fn: Callable[..., Any],
    class_map: Dict[int, str]
) -> Dict[str, Union[str, float]]:
    """
    Runs inference on a single video file.

    Returns:
        dict:
            {
                "video": filename,
                "prediction": class_label,
                "confidence": float
            }
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Import here so PyTorchVideo is only required at inference time
    from pytorchvideo.data.encoded_video import EncodedVideo

    log.info(f"Processing video: {video_path}")
    start_time = time.time()

    try:
        video = EncodedVideo.from_path(video_path)

        # Deterministic 5-second center clip for inference
        clip_duration = cfg.clip_duration
        video_meta = video.get_metadata()
        video_length = float(video_meta["video_duration"])

        # Clamp for safety
        end_sec = min(clip_duration, video_length)
        clip = video.get_clip(start_sec=0.0, end_sec=end_sec)

        if clip.get("video", None) is None:
            raise ValueError("PyTorchVideo failed to decode video.")

        # Apply your validation/production transforms
        # transform_fn expects a video tensor (clip["video"]) and returns a tensor
        video_tensor = clip["video"]
        processed = transform_fn(video_tensor)

        # Batch dimension
        tensor = processed.unsqueeze(0).to(device)

    except Exception as e:
        log.error(f"Video decoding error: {e}")
        return {"video": os.path.basename(video_path),
                "prediction": "Decoding Error",
                "confidence": 0.0}

    # Forward pass — inference mode is safest
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probabilities, 1)

    idx_int = int(idx.item())
    prediction = class_map.get(idx_int, "Unknown")
    confidence = float(conf.item())

    end_time = time.time()
    log.info(f"Inference complete in {end_time - start_time:.3f}s")

    return {
        "video": os.path.basename(video_path),
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }


# ---------------------------------------------------------------------
# CLI-Style Main Execution
# ---------------------------------------------------------------------
def main():
    """Runs inference on test videos listed in cfg or hardcoded list."""

    # Robust device handling: always prefer config.py value
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Path to model weights
    weights_path = os.path.join(cfg.output_dir, cfg.model_name, "_best_model.pth")

    # Load labels from the authoritative training CSV
    print(f"Loading class labels from: {cfg.train_csv}")
    classes = load_class_labels(cfg.train_csv)

    if len(classes) != cfg.num_classes:
        raise ValueError(f"Loaded classes count ({len(classes)}) does not match cfg.num_classes ({cfg.num_classes}).")

    # Load transforms + model
    try:
        model = load_model_for_inference(weights_path, cfg.num_classes, device)
        transform_fn = val_transform()
    except Exception:
        log.error("Failed to load model or transforms.")
        return

    # Example video list (TODO replace with CLI or API input later)
    test_videos = getattr(cfg, "inference_samples", [])
    if not test_videos:
        log.warning("No inference samples configured in cfg.inference_samples; exiting.")
        return

    log.info("----- Running Inference -----")
    for video_path in test_videos:
        try:
            result = predict_video_artifact(
                model=model,
                video_path=video_path,
                device=device,
                transform_fn=transform_fn,
                class_map=classes,
            )
            log.info("-" * 40)
            log.info(f"Video:       {result['video']}")
            log.info(f"Prediction:  {result['prediction']}")
            log.info(f"Confidence:  {result['confidence'] * 100:.2f}%")

        except FileNotFoundError as e:
            log.warning(f"Skipping video: {e}")
        except Exception as e:
            log.error(f"Unexpected inference error: {e}")

    log.info("\nInference session complete.")


if __name__ == "__main__":
    main()
