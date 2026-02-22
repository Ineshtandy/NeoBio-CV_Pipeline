"""
Legacy band detection implementations (reference, not used by default).

Kept to compare approaches or re-enable later if needed.
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


def lane_has_band_from_row_score(
    row_score_smooth: np.ndarray,
    peak_thr: float = 0.08,
    run_thr: float = 0.05,
    min_run: int = 6,
) -> bool:
    """
    Legacy: decide band presence from smoothed row-score.
    """
    peak_ok = row_score_smooth.max() > peak_thr
    above = row_score_smooth > run_thr
    run_ok = longest_true_run(above) >= min_run
    return peak_ok and run_ok


def lane_has_band_from_mask(
    lane_mask: np.ndarray,
    peak_thr: float = 0.12,
    run_thr: float = 0.08,
    min_run: int = 10,
) -> bool:
    """
    Legacy mask-only band detection.
    
    Interprets black pixels inside a lane as band candidates.
    Includes width check for band robustness.
    """
    edge_margin_px = 6
    min_width_frac = 0.30

    if lane_mask.shape[1] <= edge_margin_px * 2:
        return False

    lane_mask_trim = lane_mask[:, edge_margin_px:-edge_margin_px]
    band_pix = (lane_mask_trim == 0).astype(np.uint8)
    row_score = band_pix.mean(axis=1)
    row_score_smooth = cv2.blur(row_score.reshape(-1, 1), (1, 9)).ravel()

    width_thr = max(1, int(np.ceil(min_width_frac * band_pix.shape[1])))

    width_ok = False
    for row in band_pix:
        max_run = 0
        cur = 0
        for v in row:
            if v:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        if max_run >= width_thr:
            width_ok = True
            break

    return width_ok and lane_has_band_from_row_score(row_score_smooth, peak_thr, run_thr, min_run)


# Legacy detector registry
LEGACY_DETECTORS: dict[str, Callable[[np.ndarray], bool]] = {
    "legacy_mask": lane_has_band_from_mask,
}
