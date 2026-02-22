"""
Blot detection module - lane masks and band detection.
"""

from .lane_mask import build_lane_mask, roi_from_mask, find_vertical_separators, lanes_from_separators
from .band_detection import has_band_mask, DETECTORS

__all__ = [
    "build_lane_mask",
    "roi_from_mask",
    "find_vertical_separators",
    "lanes_from_separators",
    "has_band_mask",
    "DETECTORS",
]
