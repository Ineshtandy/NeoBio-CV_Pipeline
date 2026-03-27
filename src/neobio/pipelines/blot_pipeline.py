"""Main blot detection pipeline orchestrator."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from ..blot.lane_mask import (
    find_vertical_separators,
    lanes_from_separators,
)
from ..blot.band_detection import DETECTORS
from ..utils.debug_draw import draw_blot_debug
from ..utils.io import ensure_dir
from .shared_preprocessing import compute_shared_blot_context


def run_blot_pipeline(
    image_bgr: np.ndarray,
    band_mode: str = "mask",
    *,
    lane_mask: np.ndarray | None = None,
    roi: tuple | None = None,
    debug: bool = False,
    debug_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run complete blot detection pipeline on an image.
    
    Orchestrates:
      1. Lane mask creation
      2. ROI extraction
      3. Separator detection
      4. Lane box generation
      5. Per-lane band detection
    
    Args:
        image_bgr: Input image in BGR format
        band_mode: Band detector mode (must exist in DETECTORS)
                   Default: "mask"
        
    Returns:
        Dict with keys:
          - num_lanes (int): Number of detected lanes
          - band_present (list[bool]): Band status for each lane
          - lane_boxes (list[(x1,y1,x2,y2)]): Lane bounding boxes
          - roi (tuple): Overall ROI bounding box (x1,y1,x2,y2)
        
        Raises:
            ValueError: If band_mode not in DETECTORS
            ValueError: If image is invalid
    """
    # Validate band mode
    if band_mode not in DETECTORS:
        raise ValueError(
            f"Unknown band_mode '{band_mode}'. "
            f"Available modes: {list(DETECTORS.keys())}"
        )
    
    if lane_mask is not None and roi is not None:
        mask = lane_mask
        roi_bounds = roi
    else:
        shared_context = compute_shared_blot_context(image_bgr)
        mask = shared_context["lane_mask"]
        roi_bounds = shared_context["roi"]

    # Step 2: Extract ROI
    x1, y1, x2, y2 = roi_bounds
    
    # Step 3: Find separators in ROI space
    mask_roi = mask[y1:y2+1, x1:x2+1]
    sep_xs = find_vertical_separators(mask_roi)
    
    # Step 4: Generate lane boxes
    lane_boxes = lanes_from_separators(roi_bounds, sep_xs, pad=3)
    
    # Fallback: if no lanes detected, use entire ROI as one lane
    if not lane_boxes:
        lane_boxes = [roi_bounds]
    
    # Step 5: Run band detector on each lane
    detector = DETECTORS[band_mode]
    band_present = []
    
    for lx1, ly1, lx2, ly2 in lane_boxes:
        # Extract lane mask
        lane_mask = mask[ly1:ly2+1, lx1:lx2+1]
        
        # Detect band
        has_band = detector(lane_mask)
        band_present.append(has_band)

    if debug:
        out_dir = _make_debug_dir(debug_dir)
        _save_debug_artefacts(
            out_dir,
            image_bgr=image_bgr,
            mask=mask,
            roi=roi_bounds,
            lane_boxes=lane_boxes,
            band_present=band_present,
        )
    
    return {
        "num_lanes": len(lane_boxes),
        "band_present": band_present,
        "lane_boxes": lane_boxes,
        "roi": roi_bounds,
    }


def _make_debug_dir(debug_dir: str | None) -> str:
    root = debug_dir or "testing output images/blot_pipeline_debug"
    stamp = datetime.utcnow().strftime("blot_pipeline_%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, stamp)
    ensure_dir(out_dir)
    return out_dir


def _save_debug_artefacts(
    out_dir: str,
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    roi: tuple,
    lane_boxes: list[tuple],
    band_present: list[bool],
) -> None:
    overlay = draw_blot_debug(image_bgr, roi, lane_boxes, band_present)
    cv2.imwrite(os.path.join(out_dir, "01_full_image.png"), image_bgr)
    cv2.imwrite(os.path.join(out_dir, "02_lane_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, "03_blot_overlay.png"), overlay)
