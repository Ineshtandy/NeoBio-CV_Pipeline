"""
Google Vision OCR backend.

Runs text detection on an in-memory encoded image and normalizes
`text_annotations` into a predictable schema for downstream processing.
"""

from typing import Any

import cv2
import numpy as np


def _bbox_vertices_to_list(vertices: Any) -> list[list[int]]:
    """Convert Vision vertices to exactly four [x, y] points."""
    pts: list[list[int]] = []
    for v in list(vertices or [])[:4]:
        x = int(getattr(v, "x", 0) or 0)
        y = int(getattr(v, "y", 0) or 0)
        pts.append([x, y])

    while len(pts) < 4:
        pts.append([0, 0])

    return pts


def _normalise_text_annotations(annotations: Any) -> tuple[str, list[dict[str, Any]]]:
    """
    Normalize Google Vision `text_annotations`.

    Returns:
        (raw_text, items)
    """
    ann_list = list(annotations or [])
    if not ann_list:
        return "", []

    raw_text = str(getattr(ann_list[0], "description", "") or "")
    items: list[dict[str, Any]] = []

    for ann in ann_list[1:]:
        text = str(getattr(ann, "description", "") or "")
        vertices = getattr(getattr(ann, "bounding_poly", None), "vertices", None)
        bbox = _bbox_vertices_to_list(vertices)

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        x_min = int(min(xs))
        x_max = int(max(xs))
        y_min = int(min(ys))
        y_max = int(max(ys))
        width = int(max(0, x_max - x_min))
        height = int(max(0, y_max - y_min))

        items.append(
            {
                "text": text,
                "bbox": bbox,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "cx": (x_min + x_max) / 2.0,
                "cy": (y_min + y_max) / 2.0,
                "width": width,
                "height": height,
            }
        )

    return raw_text, items


def run_google_vision_ocr(image_bgr: np.ndarray) -> dict[str, Any]:
    """
    Run Google Vision OCR on an in-memory image.

    Args:
        image_bgr: Input image as BGR ndarray.

    Returns:
        {
            "engine": "google_vision",
            "raw_text": str,
            "items": list[dict],
        }

    Raises:
        ValueError: If input image is empty or encoding fails.
        RuntimeError: If Google Vision returns an API error.
        ImportError: If google-cloud-vision is not installed.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is empty or None.")

    ''' image currently in nd-array
        cannot send the raw numpy array directly to the Google Cloud Vision API. 
        You must use cv2.imencode to turn that array back into a "file-like" byte stream that the API understands '''
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok or encoded is None:
        raise ValueError("Failed to encode image for Google Vision OCR.")

    try:
        from google.cloud import vision
    except Exception as exc:
        raise ImportError(
            "google-cloud-vision is required for run_google_vision_ocr. "
            "Install it with: pip install google-cloud-vision"
        ) from exc

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=encoded.tobytes())
    response = client.text_detection(image=image)

    if response.error.message:
        raise RuntimeError(
            f"Google Vision OCR API error: {response.error.message}"
        )

    raw_text, items = _normalise_text_annotations(response.text_annotations)
    return {
        "engine": "google_vision",
        "raw_text": raw_text,
        "items": items,
    }
