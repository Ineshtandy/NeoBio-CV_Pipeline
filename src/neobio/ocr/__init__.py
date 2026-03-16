"""
NeoBio OCR utilities.

Provides helpers for preparing cropped image regions for downstream
OCR processing.  Does NOT perform OCR text extraction.
"""

from .region_stitching import (
    compute_top_ocr_region_bounds,
    compute_bottom_ocr_region_bounds,
    crop_region,
    stitch_regions,
)

__all__ = [
    "compute_top_ocr_region_bounds",
    "compute_bottom_ocr_region_bounds",
    "crop_region",
    "stitch_regions",
]
