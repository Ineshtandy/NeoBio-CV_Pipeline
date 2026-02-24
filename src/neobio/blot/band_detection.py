"""
MVP band detection module.

Contains the official MVP detector using mask-only approach.
"""

from typing import Callable
import cv2
import numpy as np


def longest_true_run(b: np.ndarray) -> int:
    """
    Find the longest contiguous run of True values in a boolean array.
    
    Args:
        b: Boolean or binary array
        
    Returns:
        Length of longest True run
    """
    best = 0
    cur = 0
    for v in b:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def has_band_from_row_score(
    row_score_smooth: np.ndarray,
    peak_thr: float = 0.08,
    run_thr: float = 0.05,
    min_run: int = 6,
) -> bool:
    """
    Decide band presence from smoothed row-score.
    
    Decision logic:
      - Peak must exceed peak_thr
      - Must have contiguous run of min_run rows above run_thr
    
    Args:
        row_score_smooth: Smoothed row scores (fractions 0-1)
        peak_thr: Threshold for peak detection
        run_thr: Threshold for run detection
        min_run: Minimum run length required
        
    Returns:
        True if band detected, False otherwise
    """
    peak_ok = row_score_smooth.max() > peak_thr
    above = row_score_smooth > run_thr
    run_ok = longest_true_run(above) >= min_run
    return peak_ok and run_ok


def has_band_mask(
    lane_mask: np.ndarray,
    top_crop_frac: float = 0.15,
    bottom_crop_frac: float = 0.15,
    edge_margin_px: int = 6,
    smooth_kernel_h: int = 9,
    peak_thr: float = 0.12,
    run_thr: float = 0.08,
    min_run: int = 10,
) -> bool:
    """
    MVP mask-only band detection for a single lane.
    
    Detects band presence using only the binary lane mask.
    
    Logic:
      1. Fixed crop: remove top/bottom and left/right margins
      2. Count black pixels per row (band candidates)
      3. Smooth row-scores
      4. Apply peak + run-length decision
    
    Args:
        lane_mask: uint8 binary mask for one lane (lanes white 255, bands/separators black 0)
        top_crop_frac: Fraction of lane to crop from top
        bottom_crop_frac: Fraction of lane to crop from bottom
        edge_margin_px: Margin in pixels from left/right edges
        smooth_kernel_h: Height of smoothing kernel for row scores
        peak_thr: Threshold for peak detection
        run_thr: Threshold for run-length detection
        min_run: Minimum run length required for band
        
    Returns:
        True if band detected, False otherwise
    """
    h, w = lane_mask.shape[:2]
    
    # Validate lane mask dimensions
    if w <= edge_margin_px * 2:
        return False
    
    # Compute cropping bounds
    y0 = int(h * top_crop_frac)
    y1 = int(h * (1.0 - bottom_crop_frac))
    x0 = edge_margin_px
    x1 = w - edge_margin_px
    
    # Check if cropped region is large enough
    if (y1 - y0) < 10 or (x1 - x0) < 10:
        return False
    
    # Crop the mask
    trim = lane_mask[y0:y1, x0:x1]
    
    # Band pixels: black pixels in the lane (0 value)
    band_pix = (trim == 0).astype(np.uint8)
    
    # Row score: fraction of black pixels per row
    row_score = band_pix.mean(axis=1)
    
    # Smooth row scores
    row_score_smooth = cv2.blur(row_score.reshape(-1, 1), (1, smooth_kernel_h)).ravel()
    
    # Apply decision logic
    return has_band_from_row_score(row_score_smooth, peak_thr, run_thr, min_run)


# Detector registry: maps mode names to detector functions
DETECTORS: dict[str, Callable[[np.ndarray], bool]] = {
    "mask": has_band_mask,
}
