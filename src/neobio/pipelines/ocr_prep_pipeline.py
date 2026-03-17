"""
OCR preparation pipeline.

Produces a stitched image of the regions *above* and *below* the blot ROI,
ready for downstream OCR processing.  Does NOT perform OCR text extraction.

Dependency flow
---------------
run_ocr_prep_pipeline
  → build_lane_mask          (src/neobio/blot/lane_mask.py)
  → roi_from_mask            (src/neobio/blot/lane_mask.py)
  → compute_top_ocr_region_bounds    (src/neobio/ocr/region_stitching.py)
  → compute_bottom_ocr_region_bounds (src/neobio/ocr/region_stitching.py)
  → crop_region              (src/neobio/ocr/region_stitching.py)
  → stitch_regions           (src/neobio/ocr/region_stitching.py)

The full blot pipeline (lane separation, band detection) is intentionally
NOT called here.
"""

import os
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np

from ..blot.lane_mask import build_lane_mask, roi_from_mask
from ..ocr.region_stitching import (
    compute_bottom_ocr_region_bounds,
    compute_top_ocr_region_bounds,
    crop_region,
    stitch_regions,
)
from ..utils.io import ensure_dir


def run_ocr_prep_pipeline(
    image_bgr: np.ndarray,
    top_extra_bottom_px: int = 6,
    bottom_extra_top_px: int = 6,
    gap_px: int = 20,
    gap_value: int = 255,
    orientation: str = "vertical",
    debug: bool = False,
    debug_dir: str = "ocr_prep_debug",
    input_path: Optional[str] = None,
) -> Dict:
    """
    Prepare a stitched OCR-input image from a single blot image.

    Steps
    -----
    1. Build the lane mask using the standard mask-generation logic.
    2. Extract the blot ROI from the mask.
    3. Compute bounds for the top and bottom OCR strips around the ROI.
    4. Crop both strips from the original image.
    5. Stitch them into a single image (top strip first, then bottom strip).
    6. Optionally save debug artefacts.

    Args:
        image_bgr: Input blot image in BGR format (``numpy.ndarray``).
        top_extra_bottom_px: Pixels to extend the top strip downward into
            the ROI.  ``0`` means the top strip ends exactly at ``roi_y1``.
        bottom_extra_top_px: Pixels to extend the bottom strip upward into
            the ROI.  ``0`` means the bottom strip starts exactly at
            ``roi_y2``.
        gap_px: Height of the blank separator between the two stitched
            regions (pixels).  ``0`` removes the separator.
        gap_value: Fill value for the separator strip (0–255).  Default 255
            is white.
        orientation: Stitch orientation.  Only ``"vertical"`` is supported.
        debug: If ``True``, intermediate images are saved to disk.
        debug_dir: Root directory for debug artefacts.  A per-image
            subdirectory is created inside this directory.
        input_path: Optional path to the source image file.  When provided,
            the file stem is used to name the per-image debug subdirectory.
            When omitted a UTC timestamp is used instead.

    Returns:
        Dict with keys:
            - ``"roi"`` *(tuple)*: ``(x1, y1, x2, y2)`` blot ROI.
            - ``"top_region_bounds"`` *(tuple)*: bounds of the top strip.
            - ``"bottom_region_bounds"`` *(tuple)*: bounds of the bottom strip.
            - ``"top_region_shape"`` *(tuple)*: shape of the top crop array.
            - ``"bottom_region_shape"`` *(tuple)*: shape of the bottom crop array.
            - ``"stitched_shape"`` *(tuple)*: shape of the stitched image.
            - ``"stitched_image"`` *(np.ndarray)*: the stitched output.
            - ``"debug_dir"`` *(str | None)*: path to the per-image debug
              folder when ``debug=True``, otherwise ``None``.

    Raises:
        ValueError: If ``image_bgr`` is empty or ``None``, if any
            configuration value is invalid, or if the ROI leaves no room for
            a top or bottom strip.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is empty or None.")

    if top_extra_bottom_px < 0:
        raise ValueError(
            f"top_extra_bottom_px must be >= 0, got {top_extra_bottom_px}"
        )
    if bottom_extra_top_px < 0:
        raise ValueError(
            f"bottom_extra_top_px must be >= 0, got {bottom_extra_top_px}"
        )
    if gap_px < 0:
        raise ValueError(f"gap_px must be >= 0, got {gap_px}")

    # ------------------------------------------------------------------
    # Step 1: Build lane mask (uses all library defaults; no changes)
    # ------------------------------------------------------------------
    mask = build_lane_mask(image_bgr)

    # ------------------------------------------------------------------
    # Step 2: Extract ROI
    # ------------------------------------------------------------------
    roi = roi_from_mask(mask)

    # ------------------------------------------------------------------
    # Step 3: Compute OCR strip bounds
    # ------------------------------------------------------------------
    top_bounds = compute_top_ocr_region_bounds(
        image_bgr.shape, roi, top_extra_bottom_px=top_extra_bottom_px
    )
    bottom_bounds = compute_bottom_ocr_region_bounds(
        image_bgr.shape, roi, bottom_extra_top_px=bottom_extra_top_px
    )

    # ------------------------------------------------------------------
    # Step 4: Crop strips
    # ------------------------------------------------------------------
    top_crop = crop_region(image_bgr, top_bounds)
    bottom_crop = crop_region(image_bgr, bottom_bounds)

    # ------------------------------------------------------------------
    # Step 5: Stitch
    # ------------------------------------------------------------------
    stitched = stitch_regions(
        top_crop,
        bottom_crop,
        orientation=orientation,
        gap_px=gap_px,
        gap_value=gap_value,
    )

    # ------------------------------------------------------------------
    # Step 6: Debug artefacts
    # ------------------------------------------------------------------
    per_image_debug_dir: Optional[str] = None
    if debug:
        per_image_debug_dir = _make_debug_dir(debug_dir, input_path)
        _save_debug_artefacts(
            per_image_debug_dir,
            image_bgr=image_bgr,
            mask=mask,
            top_crop=top_crop,
            bottom_crop=bottom_crop,
            stitched=stitched,
            roi=roi,
        )

    return {
        "roi": roi,
        "top_region_bounds": top_bounds,
        "bottom_region_bounds": bottom_bounds,
        "top_region_shape": top_crop.shape,
        "bottom_region_shape": bottom_crop.shape,
        "stitched_shape": stitched.shape,
        "stitched_image": stitched,
        "debug_dir": per_image_debug_dir,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_debug_dir(debug_dir: str, input_path: Optional[str]) -> str:
    """Create and return a per-image subdirectory inside *debug_dir*."""
    if input_path:
        stem = os.path.splitext(os.path.basename(input_path))[0]
    else:
        stem = datetime.utcnow().strftime("ocr_prep_%Y%m%d_%H%M%S")

    per_image_dir = os.path.join(debug_dir, stem)
    ensure_dir(per_image_dir)
    return per_image_dir


def _save_debug_artefacts(
    out_dir: str,
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    top_crop: np.ndarray,
    bottom_crop: np.ndarray,
    stitched: np.ndarray,
    roi,
) -> None:
    """Save numbered debug images to *out_dir*."""
    cv2.imwrite(os.path.join(out_dir, "01_full_image.png"), image_bgr)
    cv2.imwrite(os.path.join(out_dir, "02_lane_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, "03_top_region.png"), top_crop)
    cv2.imwrite(os.path.join(out_dir, "04_bottom_region.png"), bottom_crop)
    cv2.imwrite(os.path.join(out_dir, "05_stitched_ocr_input.png"), stitched)

    # ROI overlay — draws a green rectangle on a copy of the full image
    overlay = image_bgr.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(out_dir, "06_roi_overlay.png"), overlay)
