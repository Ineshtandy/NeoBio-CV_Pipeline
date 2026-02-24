"""
Main blot detection pipeline orchestrator.

Coordinates lane detection and band detection stages.
"""

from typing import Dict, List, Tuple
import numpy as np

from ..blot.lane_mask import (
    build_lane_mask,
    roi_from_mask,
    find_vertical_separators,
    lanes_from_separators,
)
from ..blot.band_detection import DETECTORS


def run_blot_pipeline(
    image_bgr: np.ndarray,
    band_mode: str = "mask",
) -> Dict:
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
    
    # Step 1: Build lane mask
    mask = build_lane_mask(image_bgr)
    
    # Step 2: Extract ROI
    roi = roi_from_mask(mask)
    x1, y1, x2, y2 = roi
    
    # Step 3: Find separators in ROI space
    mask_roi = mask[y1:y2+1, x1:x2+1]
    sep_xs = find_vertical_separators(mask_roi)
    
    # Step 4: Generate lane boxes
    lane_boxes = lanes_from_separators(roi, sep_xs, pad=3)
    
    # Fallback: if no lanes detected, use entire ROI as one lane
    if not lane_boxes:
        lane_boxes = [roi]
    
    # Step 5: Run band detector on each lane
    detector = DETECTORS[band_mode]
    band_present = []
    
    for lx1, ly1, lx2, ly2 in lane_boxes:
        # Extract lane mask
        lane_mask = mask[ly1:ly2+1, lx1:lx2+1]
        
        # Detect band
        has_band = detector(lane_mask)
        band_present.append(has_band)
    
    return {
        "num_lanes": len(lane_boxes),
        "band_present": band_present,
        "lane_boxes": lane_boxes,
        "roi": roi,
    }
