"""
Lane mask creation and utilities.

This module handles:
- Grey range auto-detection
- Lane mask generation from image
- ROI extraction from mask
- Vertical separator detection
- Lane box generation from separators
"""

from typing import List, Tuple
import cv2
import numpy as np


# Morphology parameters
GREY_LOW_DEFAULT = 165
GREY_HIGH_DEFAULT = 230
K_SEP_Y_HEIGHT_FRAC = 0.03
MORPH_OPEN_KERNEL = (3, 3)
MORPH_CLOSE_KERNEL_Y = None  # Computed based on image height
SEP_MIN_HEIGHT_FRAC = 0.80
SEP_MAX_WIDTH_PX = 12
SEP_DILATION_KERNEL = (3, 3)


def auto_grey_range(gray: np.ndarray) -> Tuple[int, int]:
    """
    Auto-detect grey value range from grayscale image.
    
    Ignores pure white background and very dark bands/text.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        (low, high) grey value thresholds
    """
    # Filter pixels that are likely part of the blot region
    vals = gray[(gray >= 165) & (gray <= 245)]
    if vals.size < 1000:
        # fallback if we couldn't find enough pixels
        return GREY_LOW_DEFAULT, GREY_HIGH_DEFAULT

    lo = np.percentile(vals, 10)
    hi = np.percentile(vals, 95)

    # Pad a bit for safety
    grey_low = int(max(0, lo - 15))
    grey_high = int(min(245, hi + 5))
    return grey_low, grey_high


def build_lane_mask(
    image_bgr: np.ndarray,
    debug: bool = False,
    debug_out_path: str = None,
) -> np.ndarray:
    """
    Build binary lane mask from BGR image.
    
    Lanes appear as white (255), separators/bands as black (0).
    
    Args:
        image_bgr: Input BGR image
        debug: If True, save intermediate debug images
        debug_out_path: Path to save debug mask (only if debug=True)
        
    Returns:
        uint8 mask (0/255) same size as input image
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image provided")

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Auto-detect grey range
    grey_low, grey_high = auto_grey_range(gray)
    
    # Create mask: lane region (grey) -> white (255)
    mask = cv2.inRange(gray, grey_low, grey_high)
    
    # Remove tiny white specks
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Strengthen separator lines
    inv = cv2.bitwise_not(mask)  # separators now white
    
    # Use morphology to close gaps in separators
    k_sep_y = max(25, int(round(height * K_SEP_Y_HEIGHT_FRAC)))
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_sep_y))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_sep, iterations=1)
    
    # Thicken separators slightly
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, SEP_DILATION_KERNEL)
    inv = cv2.dilate(inv, kernel_dilate, iterations=1)
    
    # Convert back: lanes white, separators black
    mask = cv2.bitwise_not(inv)
    
    if debug and debug_out_path:
        cv2.imwrite(debug_out_path, mask)
    
    return mask


def roi_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Extract region of interest (ROI) bounding box from binary mask.
    
    Finds the smallest rectangle containing all white (255) pixels.
    
    Args:
        mask: Binary mask (uint8 0/255)
        
    Returns:
        (x1, y1, x2, y2) - bounding box coordinates (inclusive)
        
    Raises:
        ValueError: If mask has no white pixels
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("Mask has no white pixels; cannot determine blot ROI.")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def find_vertical_separators(
    mask_roi: np.ndarray,
    min_height_frac: float = 0.80,
    max_width_px: int = 12,
) -> List[int]:
    """
    Detect vertical lane separators in ROI mask.
    
    Uses morphological operations to find thin vertical lines.
    
    Args:
        mask_roi: Binary mask of ROI region (lanes white, separators black)
        min_height_frac: Minimum separator height as fraction of total height
        max_width_px: Maximum separator width in pixels
        
    Returns:
        List of x-coordinates (in ROI space) of detected separators
    """
    _, bw = cv2.threshold(mask_roi, 127, 255, cv2.THRESH_BINARY)
    inv = 255 - bw  # separators now white
    
    h, _ = inv.shape
    k_h = max(20, int(h * 0.90))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h))
    vlines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(vlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sep_xs = []
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh >= int(h * min_height_frac) and ww <= max_width_px:
            sep_xs.append(x + ww // 2)
    
    sep_xs = sorted(sep_xs)
    return sep_xs


def lanes_from_separators(
    roi_bbox: Tuple[int, int, int, int],
    sep_xs: List[int],
    pad: int = 3,
    min_lane_width: int = 12,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate lane bounding boxes from separator x-coordinates.
    
    Takes detected separators and partitions ROI into lanes.
    
    Args:
        roi_bbox: ROI bounding box (x1, y1, x2, y2)
        sep_xs: List of separator x-coordinates in ROI space
        pad: Padding from separators to lane edges (pixels)
        min_lane_width: Minimum lane width to keep (pixels)
        
    Returns:
        List of lane bounding boxes [(x1, y1, x2, y2), ...]
    """
    x1, y1, x2, y2 = roi_bbox
    roi_w = x2 - x1 + 1
    
    # Create boundaries: left edge, separators, right edge
    boundaries = [0] + sep_xs + [roi_w - 1]
    
    lanes = []
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        
        # Apply padding to avoid separator regions
        lx = left + pad
        rx = right - pad
        if rx - lx + 1 < min_lane_width:
            continue
        
        # Convert back to full image coordinates
        lane_x1 = x1 + lx
        lane_x2 = x1 + rx
        lanes.append((lane_x1, y1, lane_x2, y2))
    
    return lanes
