"""
OCR pipeline orchestrator.

Builds stitched OCR input around the blot ROI, runs Google Vision OCR,
and post-processes token items into top labels and a bottom identifier.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from ..ocr.google_vision_backend import run_google_vision_ocr
from ..ocr.ocr_postprocess import (
    build_top_labels,
    join_bottom_identifier_tokens,
    split_items_by_stitch_boundary,
)
from ..ocr.region_stitching import (
    compute_bottom_ocr_region_bounds,
    compute_top_ocr_region_bounds,
    crop_region,
    rotate_image_bound,
    stitch_regions,
)
from .shared_preprocessing import compute_shared_blot_context
from ..utils.io import ensure_dir


def run_ocr_pipeline(
    image_bgr: np.ndarray,
    *,
    lane_mask: np.ndarray | None = None,
    roi: tuple | None = None,
    top_extra_bottom_px: int = 6,
    bottom_extra_top_px: int = 6,
    top_rotation_deg: float = -52.2,
    stitch_gap_px: int = 20,
    debug: bool = False,
    debug_dir: str | None = None,
) -> dict[str, Any]:
    """Run the OCR pipeline without invoking the full blot pipeline."""
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
    if stitch_gap_px < 0:
        raise ValueError(f"stitch_gap_px must be >= 0, got {stitch_gap_px}")

    if lane_mask is not None and roi is not None:
        mask = lane_mask
        roi_bounds = roi
    else:
        shared_context = compute_shared_blot_context(image_bgr)
        mask = shared_context["lane_mask"]
        roi_bounds = shared_context["roi"]

    top_bounds = compute_top_ocr_region_bounds(
        image_bgr.shape,
        roi_bounds,
        top_extra_bottom_px=top_extra_bottom_px,
    )
    bottom_bounds = compute_bottom_ocr_region_bounds(
        image_bgr.shape,
        roi_bounds,
        bottom_extra_top_px=bottom_extra_top_px,
    )

    top_crop = crop_region(image_bgr, top_bounds)
    bottom_crop = crop_region(image_bgr, bottom_bounds)

    if abs(float(top_rotation_deg)) > 1e-9:
        top_crop_rotated = rotate_image_bound(top_crop, float(top_rotation_deg))
    else:
        top_crop_rotated = top_crop

    stitched_image = stitch_regions(
        top_crop_rotated,
        bottom_crop,
        orientation="vertical",
        gap_px=stitch_gap_px,
        gap_value=255,
    )

    stitch_boundary_y = int(top_crop_rotated.shape[0] - 1)

    ocr_result = run_google_vision_ocr(stitched_image)
    raw_text = str(ocr_result.get("raw_text", ""))
    items = list(ocr_result.get("items", []))

    top_items, bottom_items = split_items_by_stitch_boundary(
        items,
        stitch_boundary_y=stitch_boundary_y,
        gap_px=stitch_gap_px,
    )

    top_labels = build_top_labels(top_items)
    bottom_identifier = join_bottom_identifier_tokens(bottom_items)

    if debug:
        out_dir = _make_debug_dir(debug_dir)
        _save_debug_artefacts(
            out_dir,
            image_bgr=image_bgr,
            top_crop=top_crop,
            top_crop_rotated=top_crop_rotated,
            bottom_crop=bottom_crop,
            stitched_image=stitched_image,
        )

    return {
        "roi": roi_bounds,
        "top_region_bounds": top_bounds,
        "bottom_region_bounds": bottom_bounds,
        "stitch_boundary_y": stitch_boundary_y,
        "top_labels": top_labels,
        "bottom_identifier": bottom_identifier,
        "top_items": top_items,
        "bottom_items": bottom_items,
        "raw_text": raw_text,
        "stitched_image": stitched_image,
    }


def _make_debug_dir(debug_dir: str | None) -> str:
    root = debug_dir or "testing output images/ocr_pipeline_debug"
    stamp = datetime.utcnow().strftime("ocr_pipeline_%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, stamp)
    ensure_dir(out_dir)
    return out_dir


def _save_debug_artefacts(
    out_dir: str,
    *,
    image_bgr: np.ndarray,
    top_crop: np.ndarray,
    top_crop_rotated: np.ndarray,
    bottom_crop: np.ndarray,
    stitched_image: np.ndarray,
) -> None:
    cv2.imwrite(os.path.join(out_dir, "01_full_image.png"), image_bgr)
    cv2.imwrite(os.path.join(out_dir, "02_top_crop.png"), top_crop)
    cv2.imwrite(os.path.join(out_dir, "03_top_crop_rotated.png"), top_crop_rotated)
    cv2.imwrite(os.path.join(out_dir, "04_bottom_crop.png"), bottom_crop)
    cv2.imwrite(os.path.join(out_dir, "05_stitched_ocr_input.png"), stitched_image)
