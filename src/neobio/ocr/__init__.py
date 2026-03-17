"""
NeoBio OCR utilities.

Provides helpers for preparing cropped image regions for downstream
OCR processing.  Does NOT perform OCR text extraction.
"""

from .region_stitching import (
    compute_top_ocr_region_bounds,
    compute_bottom_ocr_region_bounds,
    crop_region,
    rotate_image_bound,
    stitch_regions,
)
from .google_vision_backend import run_google_vision_ocr
from .ocr_postprocess import (
    split_items_by_stitch_boundary,
    group_top_items_by_line,
    sort_items_left_to_right,
    join_tokens_with_spaces,
    join_bottom_identifier_tokens,
    build_top_labels,
)

__all__ = [
    "compute_top_ocr_region_bounds",
    "compute_bottom_ocr_region_bounds",
    "crop_region",
    "rotate_image_bound",
    "stitch_regions",
    "run_google_vision_ocr",
    "split_items_by_stitch_boundary",
    "group_top_items_by_line",
    "sort_items_left_to_right",
    "join_tokens_with_spaces",
    "join_bottom_identifier_tokens",
    "build_top_labels",
]
