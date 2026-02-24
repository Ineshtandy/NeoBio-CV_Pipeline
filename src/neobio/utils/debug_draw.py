"""
Debug visualization utilities.
"""

from typing import List, Tuple
import cv2
import numpy as np


def draw_blot_debug(
    image_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
    lane_boxes: List[Tuple[int, int, int, int]],
    band_present: List[bool],
) -> np.ndarray:
    """
    Draw debug overlay on image with ROI and lane boxes.
    
    Draws:
      - ROI bounding box in green
      - Lane boxes in green (if band) or red (if white)
      - Text labels: "{i}: BAND" or "{i}: WHITE"
    
    Args:
        image_bgr: Input BGR image
        roi: ROI bounding box (x1, y1, x2, y2)
        lane_boxes: List of lane bounding boxes
        band_present: List of band presence flags (same length as lane_boxes)
        
    Returns:
        Annotated BGR image (copy)
    """
    debug = image_bgr.copy()
    
    # Draw ROI
    x1, y1, x2, y2 = roi
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw lane boxes
    for idx, (lane_box, has_band) in enumerate(zip(lane_boxes, band_present), 1):
        lx1, ly1, lx2, ly2 = lane_box
        color = (0, 255, 0) if has_band else (0, 0, 255)  # Green or Red
        cv2.rectangle(debug, (lx1, ly1), (lx2, ly2), color, 2)
        
        # Add label
        label = f"{idx}: {'BAND' if has_band else 'WHITE'}"
        cv2.putText(
            debug,
            label,
            (lx1 + 2, ly1 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    
    return debug
