"""
Integrated blot + OCR pipeline orchestrator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .blot_pipeline import run_blot_pipeline
from .ocr_pipeline import run_ocr_pipeline
from .shared_preprocessing import compute_shared_blot_context


def run_integrated_pipeline(
    image_bgr: np.ndarray,
    *,
    ocr_top_extra_bottom_px: int = 0,
    ocr_bottom_extra_top_px: int = 0,
    ocr_top_rotation_deg: float = -52.5,
    ocr_stitch_gap_px: int = 20,
    band_mode: str = "mask",
    debug: bool = False,
    debug_dir: str | None = None,
) -> dict[str, Any]:
    """Run integrated OCR and blot branches using shared preprocessing."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is empty or None.")

    shared_context = compute_shared_blot_context(image_bgr)
    lane_mask = shared_context["lane_mask"]
    roi = shared_context["roi"]

    ocr_result = run_ocr_pipeline(
        image_bgr,
        lane_mask=lane_mask,
        roi=roi,
        top_extra_bottom_px=ocr_top_extra_bottom_px,
        bottom_extra_top_px=ocr_bottom_extra_top_px,
        top_rotation_deg=ocr_top_rotation_deg,
        stitch_gap_px=ocr_stitch_gap_px,
        debug=debug,
        debug_dir=debug_dir,
    )

    blot_result = run_blot_pipeline(
        image_bgr,
        band_mode=band_mode,
        lane_mask=lane_mask,
        roi=roi,
        debug=debug,
        debug_dir=debug_dir,
    )

    top_labels = list(ocr_result.get("top_labels", []))
    band_present = list(blot_result.get("band_present", []))
    lane_boxes = list(blot_result.get("lane_boxes", []))
    bottom_identifier = str(ocr_result.get("bottom_identifier", ""))

    pair_count = min(len(top_labels), len(band_present))

    reactivity: dict[str, bool] = {}
    lanes: list[dict[str, Any]] = []
    for lane_index in range(pair_count):
        label = str(top_labels[lane_index])
        band_value = bool(band_present[lane_index])
        lane_box = lane_boxes[lane_index] if lane_index < len(lane_boxes) else None

        # Preserve first-seen mapping for duplicate labels.
        if label not in reactivity:
            reactivity[label] = band_value

        lanes.append(
            {
                "lane_index": lane_index,
                "label": label,
                "band_present": band_value,
                "lane_box": lane_box,
            }
        )

    return {
        "identifier": bottom_identifier,
        "reactivity": reactivity,
        "ocr": {
            "top_labels": top_labels,
            "bottom_identifier": bottom_identifier,
            "top_items": ocr_result.get("top_items", []),
            "bottom_items": ocr_result.get("bottom_items", []),
            "raw_text": ocr_result.get("raw_text", ""),
            "stitch_boundary_y": ocr_result.get("stitch_boundary_y"),
            "top_region_bounds": ocr_result.get("top_region_bounds"),
            "bottom_region_bounds": ocr_result.get("bottom_region_bounds"),
            "stitched_image": ocr_result.get("stitched_image"),
        },
        "blot": {
            "num_lanes": blot_result.get("num_lanes", 0),
            "lane_boxes": lane_boxes,
            "band_present": band_present,
        },
        "lanes": lanes,
        "meta": {
            "roi": roi,
            "lane_mask_shape": lane_mask.shape,
            "top_label_count": len(top_labels),
            "band_present_count": len(band_present),
            "lane_box_count": len(lane_boxes),
            "reactivity_count": len(reactivity),
        },
    }
