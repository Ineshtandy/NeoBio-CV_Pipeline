"""
Region cropping and stitching helpers for OCR preparation.

All functions are pure and stateless — they perform no I/O and hold no
state between calls.

Coordinate convention
---------------------
ROI bounding boxes follow the project-wide convention used by
``roi_from_mask``: **(x1, y1, x2, y2)** where all coordinates are
**inclusive** pixel indices in full-image space.

When converting to NumPy slice notation the inclusive upper bound must
be offset by +1::

    image[y1 : y2 + 1, x1 : x2 + 1]
"""

from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Bounds computation
# ---------------------------------------------------------------------------


def compute_top_ocr_region_bounds(
    image_shape: Tuple[int, int],
    roi: Tuple[int, int, int, int],
    top_extra_bottom_px: int = 6,
) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box for the top OCR strip.

    The top OCR region covers the full image width from the top edge down
    to ``roi_y1``, optionally extended a few pixels *into* the ROI via
    ``top_extra_bottom_px``.  All coordinates are clamped to valid image
    bounds.

    Args:
        image_shape: ``(height, width)`` of the source image.
        roi: ROI bounding box ``(x1, y1, x2, y2)`` in full-image space
             (inclusive coordinates, as returned by ``roi_from_mask``).
        top_extra_bottom_px: Number of pixels to extend the bottom edge of
            this strip downward into the ROI.  Must be >= 0.

    Returns:
        ``(x1, y1, x2, y2)`` of the top OCR strip (inclusive, clamped).

    Raises:
        ValueError: If ``top_extra_bottom_px`` is negative, if the image
            shape is invalid, or if the resulting crop is empty.
    """
    if top_extra_bottom_px < 0:
        raise ValueError(
            f"top_extra_bottom_px must be >= 0, got {top_extra_bottom_px}"
        )

    H, W = image_shape[:2]
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    _x1, roi_y1, _x2, _roi_y2 = roi

    x1 = 0
    y1 = 0
    x2 = W - 1
    y2 = min(H - 1, roi_y1 + top_extra_bottom_px)

    if y2 < y1:
        raise ValueError(
            f"Top OCR region is empty after clamping: y1={y1}, y2={y2}. "
            "The ROI starts at the very top of the image — no top strip exists."
        )

    return x1, y1, x2, y2


def compute_bottom_ocr_region_bounds(
    image_shape: Tuple[int, int],
    roi: Tuple[int, int, int, int],
    bottom_extra_top_px: int = 6,
) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box for the bottom OCR strip.

    The bottom OCR region covers the full image width from ``roi_y2`` down
    to the image bottom edge, optionally extended a few pixels *into* the
    ROI via ``bottom_extra_top_px``.  All coordinates are clamped to valid
    image bounds.

    Args:
        image_shape: ``(height, width)`` of the source image.
        roi: ROI bounding box ``(x1, y1, x2, y2)`` in full-image space
             (inclusive coordinates, as returned by ``roi_from_mask``).
        bottom_extra_top_px: Number of pixels to extend the top edge of this
            strip upward into the ROI.  Must be >= 0.

    Returns:
        ``(x1, y1, x2, y2)`` of the bottom OCR strip (inclusive, clamped).

    Raises:
        ValueError: If ``bottom_extra_top_px`` is negative, if the image
            shape is invalid, or if the resulting crop is empty.
    """
    if bottom_extra_top_px < 0:
        raise ValueError(
            f"bottom_extra_top_px must be >= 0, got {bottom_extra_top_px}"
        )

    H, W = image_shape[:2]
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    _x1, _roi_y1, _x2, roi_y2 = roi

    x1 = 0
    y1 = max(0, roi_y2 - bottom_extra_top_px)
    x2 = W - 1
    y2 = H - 1

    if y2 < y1:
        raise ValueError(
            f"Bottom OCR region is empty after clamping: y1={y1}, y2={y2}. "
            "The ROI ends at the very bottom of the image — no bottom strip exists."
        )

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def crop_region(
    image: np.ndarray,
    bounds: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Crop a rectangular region from *image* using inclusive bounds.

    Args:
        image: Source image (BGR or grayscale ``numpy.ndarray``).
        bounds: ``(x1, y1, x2, y2)`` inclusive pixel coordinates in
                full-image space.

    Returns:
        Cropped subimage as a ``numpy.ndarray`` (a view, not a copy).

    Raises:
        ValueError: If the bounds are outside the image or result in an
            empty crop.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot crop from an empty image.")

    x1, y1, x2, y2 = bounds
    H, W = image.shape[:2]

    if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
        raise ValueError(
            f"Bounds ({x1},{y1},{x2},{y2}) are out of range for image of "
            f"shape {image.shape}."
        )

    if x2 < x1 or y2 < y1:
        raise ValueError(
            f"Bounds ({x1},{y1},{x2},{y2}) produce an empty region."
        )

    # NumPy slicing uses exclusive upper bound → +1
    # [row1:row2,col1:col2]
    return image[y1 : y2 + 1, x1 : x2 + 1]


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def rotate_image_bound(
    image: np.ndarray,
    angle_deg: float,
    border_value=(255, 255, 255),
) -> np.ndarray:
    """
    Rotate an image around its center while expanding bounds to avoid cropping.

    Args:
        image: Source image (BGR or grayscale ``numpy.ndarray``).
        angle_deg: Rotation angle in degrees. Positive values rotate
            counter-clockwise.
        border_value: Fill value for exposed background. Defaults to white.

    Returns:
        Rotated image with expanded canvas.

    Raises:
        ValueError: If ``image`` is empty.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot rotate an empty image.")

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos = abs(m[0, 0])
    sin = abs(m[0, 1])

    new_w = int(round((h * sin) + (w * cos)))
    new_h = int(round((h * cos) + (w * sin)))

    # Re-center the transformed image on the expanded canvas.
    m[0, 2] += (new_w / 2.0) - cx
    m[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def stitch_regions(
    top_crop: np.ndarray,
    bottom_crop: np.ndarray,
    orientation: str = "vertical",
    gap_px: int = 20,
    gap_value: int = 255,
) -> np.ndarray:
    """
    Stitch two image crops into a single image with an optional separator gap.

    Vertical layout (default)::

        ┌──────────────┐
        │   top crop   │
        ├──────────────┤  ← gap_px rows filled with gap_value
        │ bottom crop  │
        └──────────────┘

    If the two crops have different widths, the narrower one is padded on
    the right with ``gap_value`` (white by default) to match the wider one.

    Args:
        top_crop: Upper image section (BGR or grayscale).
        bottom_crop: Lower image section (BGR or grayscale).
        orientation: Stitch direction.  Only ``"vertical"`` is supported in
            the current release.
        gap_px: Height of the blank separator strip in pixels.  Must be >= 0.
        gap_value: Pixel fill value for the separator (0–255).  Default 255
            produces a white separator.

    Returns:
        Stitched ``numpy.ndarray`` with the same dtype and channel count as
        the inputs.

    Raises:
        ValueError: If ``orientation`` is unsupported, ``gap_px`` is negative,
            or the input crops are incompatible (mismatched channel counts,
            empty arrays, or different dtypes).
    """
    if orientation != "vertical":
        raise ValueError(
            f"Unsupported orientation '{orientation}'. "
            "Only 'vertical' is supported in this release."
        )

    if gap_px < 0:
        raise ValueError(f"gap_px must be >= 0, got {gap_px}")

    for name, crop in (("top_crop", top_crop), ("bottom_crop", bottom_crop)):
        if crop is None or crop.size == 0:
            raise ValueError(f"{name} is empty.")

    if top_crop.dtype != bottom_crop.dtype:
        raise ValueError(
            f"Crop dtype mismatch: top={top_crop.dtype}, bottom={bottom_crop.dtype}."
        )

    top_channels = 1 if top_crop.ndim == 2 else top_crop.shape[2]
    bot_channels = 1 if bottom_crop.ndim == 2 else bottom_crop.shape[2]
    if top_channels != bot_channels:
        raise ValueError(
            f"Channel count mismatch: top_crop has {top_channels} channel(s), "
            f"bottom_crop has {bot_channels} channel(s)."
        )

    top_h, top_w = top_crop.shape[:2]
    bot_h, bot_w = bottom_crop.shape[:2]
    out_w = max(top_w, bot_w)

    def _pad_width(crop: np.ndarray, target_w: int) -> np.ndarray:
        """Right-pad *crop* to *target_w* columns with *gap_value*."""
        current_w = crop.shape[1]
        if current_w == target_w:
            return crop
        pad_cols = target_w - current_w
        if crop.ndim == 2:
            pad = np.full((crop.shape[0], pad_cols), gap_value, dtype=crop.dtype)
        else:
            pad = np.full(
                (crop.shape[0], pad_cols, crop.shape[2]), gap_value, dtype=crop.dtype
            )
        return np.concatenate([crop, pad], axis=1)

    top_padded = _pad_width(top_crop, out_w)
    bot_padded = _pad_width(bottom_crop, out_w)

    if gap_px > 0:
        if top_crop.ndim == 2:
            gap = np.full((gap_px, out_w), gap_value, dtype=top_crop.dtype)
        else:
            gap = np.full(
                (gap_px, out_w, top_crop.shape[2]), gap_value, dtype=top_crop.dtype
            )
        return np.concatenate([top_padded, gap, bot_padded], axis=0)

    return np.concatenate([top_padded, bot_padded], axis=0)
