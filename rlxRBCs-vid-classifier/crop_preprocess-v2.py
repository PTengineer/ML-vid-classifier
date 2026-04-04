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

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import platform

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

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
    canny_high: int = 150
) -> Tuple[int, int, int, int]:
    """
    Detect the bounding box of video content by edge detection.

    This function attempts to identify the active content area of a frame,
    effectively removing letterbox (top/bottom) or pillarbox (left/right)
    black bars using three fallback strategies: Hough Lines, Edge Density,
    and Contours.

    Args:
        frame: Input frame as numpy array (H, W, C).
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.

    Returns:
        A tuple of (x1, y1, x2, y2) representing the detected content region.
    """
    frame_height, frame_width = frame.shape[:2]

    def _valid_box(box: Tuple[int, int, int, int]) -> bool:
        """Verify the box dimensions meet minimum size requirements."""
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return False
        # Content must occupy at least 15% width and 50% height of the original dimensions
        return (
            (x2 - x1) >= frame_width * 0.15
            and (y2 - y1) >= frame_height * 0.5
        )

    def _first_spike(values: np.ndarray, threshold: float) -> Optional[int]:
        """Find the first index where values exceed the given threshold."""
        indices = np.where(values >= threshold)[0]
        return int(indices[0]) if indices.size else None

    # Pre-processing: Grayscale, Blur, and Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Close gaps in edges to create solid boundaries
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    def _sobel_box() -> Optional[Tuple[int, int, int, int]]:
        """
        Strategy 1: Detect boundaries using the Sobel gradient operator.

        This method identifies transitions in intensity by calculating the 
        gradient across the frame and searching for statistical outliers 
        (spikes) that represent the interface between letterboxing and content.
        """
        # Calculate gradients in both directions using a 5x5 Sobel kernel.
        # CV_64F is used to capture both positive (dark-to-light) 
        # and negative (light-to-dark) transitions.
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Collapse 2D gradients into 1D profiles by averaging along the axes.
        # We use the profiles to find consistent vertical and horizontal edges.
        x_profile = np.average(np.asarray(sobel_x, dtype=np.float64), axis=0)
        y_profile = np.average(np.asarray(sobel_y, dtype=np.float64), axis=1)

        # Statistical analysis for the X-axis (Width)
        x_mean, x_std = np.mean(x_profile), np.std(x_profile)
        x_max_idx = int(np.argmax(x_profile))  # Potential left edge
        x_min_idx = int(np.argmin(x_profile))  # Potential right edge

        # Statistical analysis for the Y-axis (Height)
        y_mean, y_std = np.mean(y_profile), np.std(y_profile)
        y_max_idx = int(np.argmax(y_profile))  # Potential top edge
        y_min_idx = int(np.argmin(y_profile))  # Potential bottom edge

        # Threshold for a "significant" edge (3 standard deviations from mean)
        std_factor = 3

        # Determine X boundaries based on statistical significance
        if x_profile[x_max_idx] > x_mean + (std_factor * x_std):
            x_left = x_max_idx
        else:
            x_left = 0

        if x_profile[x_min_idx] < x_mean - (std_factor * x_std):
            x_right = x_min_idx
        else:
            x_right = frame_width

        # Determine Y boundaries based on statistical significance
        if y_profile[y_max_idx] > y_mean + (std_factor * y_std):
            y_top = y_max_idx
        else:
            y_top = 0

        if y_profile[y_min_idx] < y_mean - (std_factor * y_std):
            y_bottom = y_min_idx
        else:
            y_bottom = frame_height

        # Validate orientation: Ensure coordinates are not inverted.
        # This acts as a safety reset if noise creates a false-positive spike.
        if x_left >= x_right:
            x_left, x_right = 0, frame_width
        if y_top >= y_bottom:
            y_top, y_bottom = 0, frame_height

        # Final conversion to integer tuple for the crop bounding box
        box = (int(x_left), int(y_top), int(x_right), int(y_bottom))
        
        return box if _valid_box(box) else None

    def _hough_box() -> Optional[Tuple[int, int, int, int]]:
        """Strategy 2: Detect boundaries using Probabilistic Hough Transform."""
        min_line_len_x = max(int(frame_width * 0.5), 20)
        min_line_len_y = max(int(frame_height * 0.5), 20)
        threshold = max(int(frame_width * 0.15), 80)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_len_x,
            maxLineGap=20,
        )
        if lines is None:
            return None

        lefts, rights, tops, bottoms = [], [], [], []
        
        # Define search bands (outer 23% of the frame sides, 8% of the top/bottom)
        left_band = int(frame_width * 0.23)
        right_band = frame_width - left_band
        top_band = int(frame_height * 0.08)
        bottom_band = frame_height - top_band

        # normalize to shape (N, 4)
        lines = np.squeeze(lines)            # -> (N,4) if shape was (N,1,4)
        lines = lines.reshape(-1, 4)         # safe fallback if squeeze left (4,) when N==1        

        for x1, y1, x2, y2 in lines:
            # line is returned as a list containing [x1, y1, x2, y2]
            dx, dy = x2 - x1, y2 - y1

            # Categorize horizontal lines (tops/bottoms)
            if abs(dy) <= 10 and abs(dx) >= min_line_len_x * 0.7:  # 70% of min length
                y_avg = int((y1 + y2) / 2)
                if y_avg <= top_band:
                    tops.append(y_avg)
                elif y_avg >= bottom_band:
                    bottoms.append(y_avg)
            
            # Categorize vertical lines (lefts/rights)
            elif abs(dx) <= 10 and abs(dy) >= min_line_len_y * 0.7:  # 70% of min length
                x_avg = int((x1 + x2) / 2)
                if x_avg <= left_band:
                    lefts.append(x_avg)
                elif x_avg >= right_band:
                    rights.append(x_avg)

        if not (lefts and rights and tops and bottoms):
            return None

        # Use percentiles to find conservative boundaries among outliers
        return (
            int(np.percentile(lefts, 85)),
            int(np.percentile(tops, 10)),
            int(np.percentile(rights, 15)),
            int(np.percentile(bottoms, 90)),
        )

    def _density_box() -> Optional[Tuple[int, int, int, int]]:
        """Strategy 3: Detect boundaries via pixel intensity projections."""
        col_sum = np.sum(edges, axis=0).astype(np.float32)
        row_sum = np.sum(edges, axis=1).astype(np.float32)

        left_band = max(int(frame_width * 0.23), 2)
        top_band = max(int(frame_height * 0.08), 2)
        right_band = frame_width - left_band
        bottom_band = frame_height - top_band

        # Calculate adaptive thresholds based on regional means
        def get_thresh(data):
            return max(np.mean(data) * 3.0, np.percentile(data, 80), 10.0)

        l_thresh = get_thresh(col_sum[:left_band])
        r_thresh = get_thresh(col_sum[right_band:])
        t_thresh = get_thresh(row_sum[:top_band])
        b_thresh = get_thresh(row_sum[bottom_band:])

        # Detect the first significant edge transition from each side with an inset
        inset = int(frame_width * 0.13)
        left = _first_spike(col_sum[inset:left_band], l_thresh)
        if left is not None: left += inset
        right_rel = _first_spike(col_sum[right_band:][::-1], r_thresh)
        top = _first_spike(row_sum[:top_band], t_thresh)
        bottom_rel = _first_spike(row_sum[bottom_band:][::-1], b_thresh)

        if left is None or right_rel is None or top is None or bottom_rel is None:
            return None

        right = frame_width - right_rel - 1
        bottom = frame_height - bottom_rel - 1
        
        return (
            left, 
            top, 
            right, 
            bottom
        )

    def _contour_box() -> Tuple[int, int, int, int]:
        """Strategy 4: Fallback using the largest detected external contour."""
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0, 0, frame_width, frame_height

        candidates = []
        frame_area = frame_width * frame_height
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # Filter for reasonably sized content blocks, must be less than 90% of total area 
            if w >= frame_width * 0.2 and h >= frame_height * 0.2 and area < frame_area * 0.90:
                candidates.append((x, y, x + w, y + h))

        if not candidates:
            return 0, 0, frame_width, frame_height

        # Return the candidate with the largest area
        return max(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    # Execute strategies in order of reliability 
    for strategy in (_sobel_box, _hough_box, _density_box, _contour_box):
        candidate = strategy()
        if candidate and _valid_box(candidate):
            return candidate
    
    # Fallback to full frame if no valid crop region is detected
    return (0, 0, frame_width, frame_height)
    

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
     the intersection of all detected content regions to ensure no active 
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
            if (res[2] - res[0]) > frame_width * 0.2 and \
               (res[3] - res[1]) > frame_height * 0.35:
                crop_regions.append(res)
        except Exception as e:
            if logger_instance:
                logger_instance.warning(f"Detection failed at frame {idx}: {e}")
        
    cap.release()

    if not crop_regions:
        return 0, 0, frame_width, frame_height

    # Compute consensus: Using percentiles for robust boundary detection
    x1 = int(np.percentile([box[0] for box in crop_regions], 15))
    y1 = int(np.median([box[1] for box in crop_regions]))
    x2 = int(np.percentile([box[2] for box in crop_regions], 85))
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
