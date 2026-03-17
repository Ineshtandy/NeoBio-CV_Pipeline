"""
OCR post-processing helpers for stitched OCR output.
"""

from __future__ import annotations

import string
from statistics import median
from typing import Any


def split_items_by_stitch_boundary(
    items,
    stitch_boundary_y,
    gap_px=0,
) -> tuple[list, list]:
    """
    Split OCR token items into top and bottom sets using token center-y.

    Rules:
    - `cy <= stitch_boundary_y`         -> top
    - `cy > stitch_boundary_y + gap_px` -> bottom
    - inside the separator gap          -> ignored
    """
    top: list = []
    bottom: list = []

    gap = max(0, int(gap_px))
    boundary = float(stitch_boundary_y)
    bottom_start = boundary + gap

    for item in list(items or []):
        cy = float(item.get("cy", 0.0))
        if cy <= boundary:
            top.append(item)
        elif cy > bottom_start:
            bottom.append(item)

    return top, bottom


def group_top_items_by_line(top_items, line_threshold_px=None) -> list[list[dict]]:
    """
    Group top OCR token items by visual text lines using center-y proximity.

    If `line_threshold_px` is None, derive a threshold from median token
    height with a small lower bound for stability.
    """
    items = sorted(list(top_items or []), key=lambda it: float(it.get("cy", 0.0)))
    if not items:
        return []

    if line_threshold_px is None:
        heights = [max(1.0, float(it.get("height", 0.0))) for it in items]
        line_threshold = max(4.0, float(median(heights)) * 0.6)
    else:
        line_threshold = max(1.0, float(line_threshold_px))

    lines: list[list[dict[str, Any]]] = []
    current_line: list[dict[str, Any]] = [items[0]]
    current_cy = float(items[0].get("cy", 0.0))

    for item in items[1:]:
        cy = float(item.get("cy", 0.0))
        if abs(cy - current_cy) <= line_threshold:
            current_line.append(item)
            current_cy = (
                current_cy * (len(current_line) - 1) + cy
            ) / len(current_line)
        else:
            lines.append(current_line)
            current_line = [item]
            current_cy = cy

    lines.append(current_line)
    return lines


def sort_items_left_to_right(items) -> list[dict]:
    """Sort OCR token items by x_min ascending."""
    return sorted(list(items or []), key=lambda it: float(it.get("x_min", 0.0)))


def join_tokens_with_spaces(items) -> str:
    """Join OCR token texts into a single space-separated label."""
    tokens = [str(it.get("text", "")).strip() for it in list(items or [])]
    return " ".join(tok for tok in tokens if tok)


def _is_punct_token(token: str) -> bool:
    return bool(token) and all(ch in string.punctuation for ch in token)


def join_bottom_identifier_tokens(items) -> str:
    """
    Join bottom identifier tokens with punctuation-aware spacing.

    Punctuation tokens attach directly to adjacent tokens.
    """
    ordered = sort_items_left_to_right(items)
    tokens = [str(it.get("text", "")).strip() for it in ordered]
    tokens = [t for t in tokens if t]

    if not tokens:
        return ""

    out = tokens[0]
    prev = tokens[0]
    for tok in tokens[1:]:
        if _is_punct_token(tok) or _is_punct_token(prev):
            out += tok
        else:
            out += f" {tok}"
        prev = tok

    return out


def build_top_labels(top_items) -> list[str]:
    """
    Build top labels ordered top-to-bottom by grouping items into lines,
    sorting each line left-to-right, and joining with spaces.
    """
    labels: list[str] = []
    for line_items in group_top_items_by_line(top_items):
        ordered_line = sort_items_left_to_right(line_items)
        label = join_tokens_with_spaces(ordered_line)
        if label:
            labels.append(label)
    return labels
