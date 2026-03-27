"""
Shared preprocessing helpers for blot-related pipelines.
"""

from __future__ import annotations

import numpy as np

from ..blot.lane_mask import build_lane_mask, roi_from_mask


def compute_shared_blot_context(
    image_bgr: np.ndarray,
) -> dict:
    """Compute shared lane mask and ROI once for downstream pipelines."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is empty or None.")

    lane_mask = build_lane_mask(image_bgr)
    roi = roi_from_mask(lane_mask)
    return {
        "lane_mask": lane_mask,
        "roi": roi,
    }
