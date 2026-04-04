"""
crop_preprocess.py — Preprocessing script to remove black bars from videos

Purpose:
- Scan raw videos in data/unprocessed/
- Detect and remove letterbox/side black bars using edge detection
- Output cropped videos with "-crop" suffix to data/videos/
- Preserve original fps and codec for minimal distortion
- Log processing details and dimension statistics

Usage:
    python crop_preprocess.py
    python crop_preprocess.py --input_dir data/unprocessed --output_dir data/videos --canny_threshold_low 50 --canny_threshold_high 150
"""

from __future__ import annotations
import argparse
import cv2
import numpy as np
import platform
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

# Constants
SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
LOG_NAME = None  # Will be set in main
LOG_FILE = 'crop-proc.txt'
log = None  # Will be initialized in main

# ---------------------------------------------------------
# Crop Detection Functions
# ---------------------------------------------------------

def detect_crop_region(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    brightness_threshold: int = 15
) -> Tuple[int, int, int, int]:
    """
    Detect the bounding box of video content using brightness analysis with
    morphological cleaning to remove thin artifacts (gradient bars, lines).
    
    This function creates a binary mask of bright areas, applies morphological
    operations to remove thin graphics, then finds the largest solid content region.
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame as BGR color array with shape (H, W, 3).
    canny_low : int, optional
        Unused (kept for API compatibility). Default is 50.
    canny_high : int, optional
        Unused (kept for API compatibility). Default is 150.
    brightness_threshold : int, optional
        Grayscale intensity threshold (0-255) for initial content detection.
        Default is 30. Increase for videos with bright bars; decrease for dark.
    
    Returns:
    --------
    Tuple[int, int, int, int]
        Bounding box (x1, y1, x2, y2) of detected content region.
        Returns full frame (0, 0, width, height) if detection fails.
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Convert to grayscale for brightness analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray, dtype=np.uint8)
    
    # Create binary mask: bright pixels above threshold
    _, binary_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # Remove thin artifacts (gradient bars, lines) using morphological opening.
    # Use horizontal and vertical kernels separately to target bar orientations.
    VLINE_THICK = 45
    HLINE_THICK = 5
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (HLINE_THICK, 1))  # Remove horizontal lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, VLINE_THICK))  # Remove vertical lines
    
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_h)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_v)
    
    # Close small gaps within content using morphological closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Find contours in the cleaned binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0, frame_width, frame_height)
    
    # Select the largest contour (should be the main content region)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Validate minimum content size constraints
    content_width: int = x2 - x1
    content_height: int = y2 - y1
    min_width: float = frame_width * 0.25
    min_height: float = frame_height * 0.5
    
    # Return full frame if detected region is too small
    if content_width < min_width or content_height < min_height:
        return (0, 0, frame_width, frame_height)
    
    return (int(x1), int(y1), int(x2), int(y2))


def finalize_even_coordinates(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
    """
    Ensure the bounding box coordinates result in even width and height.
    Rounding inward (increasing x1/y1 and decreasing x2/y2) avoids black edges.
    """
    # Round inward to the nearest even integer
    # x1: Round up to even
    final_x1 = int(np.ceil(x1 / 2) * 2)
    # y1: Round up to even
    final_y1 = int(np.ceil(y1 / 2) * 2)
    # x2: Round down to even
    final_x2 = int(np.floor(x2 / 2) * 2)
    # y2: Round down to even
    final_y2 = int(np.floor(y2 / 2) * 2)

    # Final Safety Check: Ensure width and height are still valid/positive
    if (final_x2 - final_x1) <= 0 or (final_y2 - final_y1) <= 0:
         # Fallback logic if inward rounding collapsed the box
         return (int(x1), int(y1), int(x2), int(y2))

    return final_x1, final_y1, final_x2, final_y2


def compute_crop_consensus(
    video_path: str,
    num_sample_frames: int = 10,
    canny_low: int = 50,
    canny_high: int = 150,
    margin_px: int = 3,
    logger_instance = None,
) -> Tuple[int, int, int, int]:
    """
    Analyze multiple video frames to find a consensus cropping region.

    This function samples frames across the video duration and calculates
     the median of all detected content regions to ensure no active
    pixels are accidentally cropped.

    Args:
        video_path: Path to the input video file.
        num_sample_frames: Total frames to sample for analysis.
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.
        margin_px: Pixels to shrink the final box outward for safety.
        logger_instance: Optional logger for warning messages.

    Returns:
        A conservative bounding box (x1, y1, x2, y2).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Spread samples evenly across the video
    num_samples = min(num_sample_frames, frame_count)
    sample_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)

    crop_regions = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue

        try:
            res = detect_crop_region(frame, canny_low, canny_high)
            # Ensure the detected region is significant
            if (res[2] - res[0]) > frame_width * 0.25 and \
               (res[3] - res[1]) > frame_height * 0.45:
                crop_regions.append(res)
        except Exception as e:
            if logger_instance:
                logger_instance.warning(f"Detection failed at frame {idx}: {e}")

    cap.release()

    if not crop_regions:
        return 0, 0, frame_width, frame_height

    # Compute consensus: Using percentiles for robust boundary detection
    x1 = int(np.percentile([box[0] for box in crop_regions], 20))  # Lower percentile for left to crop more
    y1 = int(np.median([box[1] for box in crop_regions]))
    x2 = int(np.percentile([box[2] for box in crop_regions], 80))  # Higher for right
    y2 = int(np.median([box[3] for box in crop_regions]))       

    # Apply safety margin and ensure coordinates remain within frame bounds
    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(frame_width, x2 + margin_px)
    y2 = min(frame_height, y2 + margin_px)

    return finalize_even_coordinates(int(x1), int(y1), int(x2), int(y2))

# ---------------------------------------------------------
# Video Cropping and Writing
# ---------------------------------------------------------

def crop_video(
    input_path: str,
    output_path: str,
    canny_low: int = 50,
    canny_high: int = 150,
    margin_px: int = 3,
    sample_frames: int = 10,
    logger_instance = None,
) -> Dict[str, Any]:
    """
    Crop video to remove black bars and save to output path.
    Preserves original fps and codec.

    Args:
        input_path: Path to input video.
        output_path: Path to output cropped video.
        canny_low: Lower Canny threshold.
        canny_high: Upper Canny threshold.
        margin_px: Safety margin around detected content.
        sample_frames: Number of frames to analyze for crop consensus.

    Returns:
        Dictionary with processing results:
        {
            'success': bool,
            'input_dims': (width, height),
            'output_dims': (width, height),
            'fps': float,
            'frames_written': int,
            'error': str or None,
        }
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {
                'success': False,
                'input_dims': None,
                'output_dims': None,
                'fps': None,
                'frames_written': 0,
                'error': f'Cannot open video: {input_path}',
            }

        # Read video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        input_dims = (frame_width, frame_height)

        # Compute crop region from sample frames
        x1, y1, x2, y2 = compute_crop_consensus(
            input_path,
            num_sample_frames=sample_frames,
            canny_low=canny_low,
            canny_high=canny_high,
            margin_px=margin_px,
            logger_instance=logger_instance,
        )

        # Ensure valid crop box
        if x2 <= x1 or y2 <= y1:
            cap.release()
            return {
                'success': False,
                'input_dims': input_dims,
                'output_dims': None,
                'fps': fps,
                'frames_written': 0,
                'error': 'Invalid crop region detected',
            }

        output_width = x2 - x1
        output_height = y2 - y1
        output_dims = (output_width, output_height)

        # Create video writer with same codec/fps
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (output_width, output_height),
        )

        if not out.isOpened():
            cap.release()
            return {
                'success': False,
                'input_dims': input_dims,
                'output_dims': output_dims,
                'fps': fps,
                'frames_written': 0,
                'error': f'Cannot create output video writer',
            }

        # Reset to beginning and write cropped frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_written = 0

        for fr in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame
            cropped = frame[y1:y2, x1:x2]
            out.write(cropped)
            frames_written += 1

        cap.release()
        out.release()

        return {
            'success': True,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'fps': fps,
            'frames_written': frames_written,
            'error': None,
        }

    except Exception as e:
        return {
            'success': False,
            'input_dims': None,
            'output_dims': None,
            'fps': None,
            'frames_written': 0,
            'error': str(e),
        }

# ---------------------------------------------------------
# Statistics and Reporting
# ---------------------------------------------------------

def compute_statistics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute min/max dimensions and summary statistics.

    Args:
        results: List of result dictionaries from crop_video.

    Returns:
        Dictionary with computed statistics.
    """
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if not successful:
        return {
            'total_processed': len(results),
            'total_success': 0,
            'total_failed': len(failed),
            'min_output_dims': None,
            'max_output_dims': None,
            'min_fps': None,
            'max_fps': None,
        }

    output_dims_list = [r['output_dims'] for r in successful if r['output_dims']]
    fps_list = [r['fps'] for r in successful if r['fps']]

    if output_dims_list:
        areas = [w * h for w, h in output_dims_list]
        min_dims = output_dims_list[areas.index(min(areas))]
        max_dims = output_dims_list[areas.index(max(areas))]
    else:
        min_dims = max_dims = None

    min_fps = min(fps_list) if fps_list else None
    max_fps = max(fps_list) if fps_list else None

    return {
        'total_processed': len(results),
        'total_success': len(successful),
        'total_failed': len(failed),
        'min_output_dims': min_dims,
        'max_output_dims': max_dims,
        'min_fps': min_fps,
        'max_fps': max_fps,
    }


# ---------------------------------------------------------
# Main Processing
# ---------------------------------------------------------

def main() -> None:
    """Entry point for video crop preprocessing."""
    
    # Initialize logger
    import logger
    global log, LOG_NAME
    LOG_NAME = f'{platform.node()}-crop-preprocess'
    log = logger.get_logger(LOG_NAME, LOG_FILE)
    
    parser = argparse.ArgumentParser(
        description="Preprocess videos by removing black bars (letterbox/sidebars)"
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path('data/unprocessed'),
        help='Path to raw video directory (default: data/unprocessed)',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/videos'),
        help='Path to output cropped video directory (default: data/videos)',
    )
    parser.add_argument(
        '--canny_threshold_low',
        type=int,
        default=50,
        help='Lower threshold for Canny edge detection (default: 50)',
    )
    parser.add_argument(
        '--canny_threshold_high',
        type=int,
        default=150,
        help='Upper threshold for Canny edge detection (default: 150)',
    )
    parser.add_argument(
        '--margin_px',
        type=int,
        default=3,
        help='Safety margin (pixels) around detected content (default: 3)',
    )
    parser.add_argument(
        '--sample_frames',
        type=int,
        default=10,
        help='Number of frames to analyze for crop consensus (default: 10)',
    )
    parser.add_argument(
        '--sample_every',
        type=int,
        default=2,
        help='Log 1 of every N successful videos (default: 2)',
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    canny_low = args.canny_threshold_low
    canny_high = args.canny_threshold_high
    margin_px = args.margin_px
    sample_frames = args.sample_frames
    sample_every = args.sample_every

    # Validate input directory
    if not input_dir.exists():
        log.error(f"Input directory not found: {input_dir}")
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting video crop preprocessing")
    log.info(f"Input: {input_dir.resolve()}")
    log.info(f"Output: {output_dir.resolve()}")
    log.info(f"Canny thresholds: {canny_low}-{canny_high}, Margin: {margin_px}px, Sample frames: {sample_frames}")

    # Scan input directory for video files
    video_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    )

    if not video_files:
        log.warning(f"No video files found in {input_dir}")
        print(f"Warning: No video files found in {input_dir}")
        return

    log.info(f"Found {len(video_files)} video(s) to process")

    # Process each video
    results = []
    total_videos = len(video_files)

    for i, video_file in enumerate(video_files, 1):
        video_name = video_file.name
        output_name = f"{video_file.stem}-crop{video_file.suffix}"
        output_path = output_dir / output_name

        log.info(f"[{i}/{len(video_files)}] Processing: {video_name}")

        result = crop_video(
            str(video_file),
            str(output_path),
            canny_low=canny_low,
            canny_high=canny_high,
            margin_px=margin_px,
            sample_frames=sample_frames,
            logger_instance=log,
        )

        results.append(result)

        if result['success']:
            input_w, input_h = result['input_dims']
            output_w, output_h = result['output_dims']
            # Sample logging: log 1 of every N successful videos
            if i % sample_every == 0 or i == 1 or i == total_videos:
                log.info(
                    f"  ✓ Success: {video_name} "
                    f"({input_w}×{input_h} → {output_w}×{output_h}, {result['fps']:.1f} fps, "
                    f"{result['frames_written']} frames)"
                )
            print(f"  ✓ {video_name} → {output_name}")
        else:
            log.error(f"  ✗ Failed: {video_name} — {result['error']}")
            print(f"  ✗ {video_name} — Error: {result['error']}")

    # Compute and log statistics
    stats = compute_statistics(results)

    log.info("")
    log.info("=" * 70)
    log.info("CROP PREPROCESSING SUMMARY")
    log.info("=" * 70)
    log.info(f"Total videos processed: {stats['total_success']} success, {stats['total_failed']} failed")

    if stats['total_success'] > 0:
        log.info(f"Output dimensions:")
        log.info(f"  Min: {stats['min_output_dims']} px")
        log.info(f"  Max: {stats['max_output_dims']} px")
        log.info(f"FPS range: {stats['min_fps']:.1f} - {stats['max_fps']:.1f}")

    log.info("=" * 70)

    print(f"\nProcessing complete. {stats['total_success']}/{len(results)} videos successfully cropped.")
    print(f"Details written to logs/{LOG_FILE}")


if __name__ == "__main__":
    main()
