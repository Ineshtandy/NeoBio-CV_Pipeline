#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


GREY_LOW = 165
GREY_HIGH = 230

def auto_grey_range(gray: np.ndarray):
    # ignore pure white background and very dark bands/text
    vals = gray[(gray >= 165) & (gray <= 245)]
    if vals.size < 1000:
        # fallback if we couldn't find enough pixels
        return 165, 230

    lo = np.percentile(vals, 10)
    hi = np.percentile(vals, 95)

    # pad a bit
    GREY_LOW = int(max(0, lo - 15))
    GREY_HIGH = int(min(245, hi + 5))
    return GREY_LOW, GREY_HIGH



def _find_lane_boundaries(v_proj: np.ndarray, min_width: int = 20) -> List[Tuple[int, int]]:
    """
    Find lane column ranges from vertical projection.
    Returns list of (x_start, x_end) tuples for each detected lane.
    """
    threshold = np.max(v_proj) * 0.15
    above_threshold = v_proj > threshold
    
    lanes = []
    in_lane = False
    lane_start = 0
    
    for col in range(len(v_proj)):
        if above_threshold[col]:
            if not in_lane:
                lane_start = col
                in_lane = True
        else:
            if in_lane:
                lane_width = col - lane_start
                if lane_width >= min_width:
                    lanes.append((lane_start, col))
                in_lane = False
    
    if in_lane:
        lane_width = len(v_proj) - lane_start
        if lane_width >= min_width:
            lanes.append((lane_start, len(v_proj)))
    
    print(lanes)
    return lanes


def _find_grey_extent_in_lane(mask: np.ndarray, x_start: int, x_end: int) -> Tuple[int, int]:
    """
    Find top and bottom row bounds of grey region within a lane column range.
    Returns (y_top, y_bottom).
    """
    lane_region = mask[:, x_start:x_end]
    h_proj = np.sum(lane_region, axis=1)
    
    threshold = np.max(h_proj) * 0.1
    above_threshold = h_proj > threshold
    
    if not np.any(above_threshold):
        return (0, mask.shape[0])
    
    rows_with_grey = np.where(above_threshold)[0]
    y_top = int(rows_with_grey[0])
    y_bottom = int(rows_with_grey[-1])
    
    return (y_top, y_bottom)


def find_roi_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("Mask has no white pixels; cannot determine blot ROI.")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def find_vertical_separators(
    mask_roi: np.ndarray,
    min_height_frac: float = 0.80,
    max_width_px: int = 12,
) -> List[int]:
    _, bw = cv2.threshold(mask_roi, 127, 255, cv2.THRESH_BINARY)
    inv = 255 - bw

    h, _ = inv.shape
    k_h = max(20, int(h * 0.90))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h))
    vlines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(vlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sep_xs = []
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh >= int(h * min_height_frac) and ww <= max_width_px:
            sep_xs.append(x + ww // 2)

    sep_xs = sorted(sep_xs)
    return sep_xs


def lane_ranges_from_separators(
    roi_bbox: Tuple[int, int, int, int],
    sep_xs: List[int],
    pad: int = 3,
    min_lane_width: int = 12,
) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi_bbox
    roi_w = x2 - x1 + 1

    boundaries = [0] + sep_xs + [roi_w - 1]

    lanes = []
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]

        lx = left + pad
        rx = right - pad
        if rx - lx + 1 < min_lane_width:
            continue

        lane_x1 = x1 + lx
        lane_x2 = x1 + rx
        lanes.append((lane_x1, y1, lane_x2, y2))
    return lanes


def longest_true_run(b: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in b:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def lane_has_band_from_row_score(
    row_score_smooth: np.ndarray,
    peak_thr: float = 0.08,
    run_thr: float = 0.05,
    min_run: int = 6,
) -> bool:
    peak_ok = row_score_smooth.max() > peak_thr
    above = row_score_smooth > run_thr
    run_ok = longest_true_run(above) >= min_run
    return peak_ok and run_ok


def lane_has_band_from_mask(
    lane_mask: np.ndarray,
    peak_thr: float = 0.12,
    run_thr: float = 0.08,
    min_run: int = 10,
) -> bool:
    """
    Mask-only band detection.
    Interprets black pixels inside a lane as band candidates.
    """
    edge_margin_px = 6
    min_width_frac = 0.30

    if lane_mask.shape[1] <= edge_margin_px * 2:
        return False

    lane_mask_trim = lane_mask[:, edge_margin_px:-edge_margin_px]
    band_pix = (lane_mask_trim == 0).astype(np.uint8)
    row_score = band_pix.mean(axis=1)
    row_score_smooth = cv2.blur(row_score.reshape(-1, 1), (1, 9)).ravel()

    width_thr = max(1, int(np.ceil(min_width_frac * band_pix.shape[1])))

    width_ok = False
    for row in band_pix:
        max_run = 0
        cur = 0
        for v in row:
            if v:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        if max_run >= width_thr:
            width_ok = True
            break

    return width_ok and lane_has_band_from_row_score(row_score_smooth, peak_thr, run_thr, min_run)


def build_lane_mask(image: np.ndarray) -> np.ndarray:
    # checking valid image
    if image is None or image.size == 0:
        raise ValueError("Empty image provided")

    # converting to greyscale + noting dimensions
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('./gray_image.png',gray)
    GREY_LOW, GREY_HIGH = auto_grey_range(gray)

    mask = cv2.inRange(gray, GREY_LOW, GREY_HIGH)

    # cv2.imwrite('./mask_prior_morph.png',mask)


    # k = max(3, int(round(min(height, width) * 0.008)))
    # if k % 2 == 0:
    #     k += 1
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # mask is from inRange: lanes are white, separators are black

    # # (optional) remove tiny white specks first
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # # --- strengthen / connect separator lines ---
    inv = cv2.bitwise_not(mask)  # now separators are white

    k_sep_y = max(25, int(round(height * 0.03)))  # try 0.03–0.08 of height
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_sep_y))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_sep, iterations=1)

    # # (optional) thicken separators slightly so projection valleys are stronger
    inv = cv2.dilate(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    mask = cv2.bitwise_not(inv)  # back to original meaning: lanes white, separators black


    cv2.imwrite('./mask_post_morph.png',mask)

    return mask


def detect_lane_bands(image_bgr: np.ndarray, band_mode: str = "grayscale") -> dict:
    mask = build_lane_mask(image_bgr)
    roi = find_roi_from_mask(mask)

    x1, y1, x2, y2 = roi
    mask_roi = mask[y1:y2+1, x1:x2+1]
    sep_xs = find_vertical_separators(mask_roi)

    lane_boxes = lane_ranges_from_separators(roi, sep_xs, pad=3)

    # band_mode usage:
    # - "grayscale" (default): uses grayscale lane crop + Otsu, then masks it
    # - "mask": uses mask-only band detection (no grayscale)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    band_present = []
    lane_scores = []
    for (lx1, ly1, lx2, ly2) in lane_boxes:
        lane_gray = gray[ly1:ly2+1, lx1:lx2+1]
        lane_mask = mask[ly1:ly2+1, lx1:lx2+1]

        if band_mode == "mask":
            # Mask-only mode: use binary mask to decide band presence
            has_band = lane_has_band_from_mask(lane_mask)
            row_score_smooth = None
        else:
            # Grayscale mode: Otsu threshold to get dark pixels
            _, dark = cv2.threshold(lane_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Mask dark pixels: keep only dark pixels inside the lane mask region
            dark_masked = cv2.bitwise_and(dark, lane_mask)

            # Compute row-score from masked dark pixels (only inside blot region)
            row_score = (dark_masked > 0).mean(axis=1)
            row_score_smooth = cv2.blur(row_score.reshape(-1, 1), (1, 9)).ravel()

            has_band = lane_has_band_from_row_score(row_score_smooth)
        band_present.append(has_band)
        if row_score_smooth is None:
            lane_scores.append(0.0)
        else:
            lane_scores.append(float(row_score_smooth.max()))

    return {
        "roi": roi,
        "lane_boxes": lane_boxes,
        "band_present": band_present,
        "lane_peak_scores": lane_scores,
        "separator_xs_roi": sep_xs,
    }


def _draw_debug(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    debug = image.copy()
    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return debug


def _default_debug_path(image_path: str) -> str:
    base, ext = os.path.splitext(image_path)
    if not ext:
        ext = ".png"
    return f"./testing output images/{base.split('/')[1]}_debug_blot.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect grey blot rectangle and save debug image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--debug-out", help="Optional debug output path")
    parser.add_argument(
        "--band-mode",
        choices=["grayscale", "mask"],
        default="grayscale",
        help="Band detection mode: grayscale (default) or mask-only.",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    # Example: python blot_detect.py <image> --band-mode mask
    result = detect_lane_bands(image, band_mode=args.band_mode)
    roi = result["roi"]
    lane_boxes = result["lane_boxes"]
    band_present = result["band_present"]
    band_scores = result['lane_peak_scores']

    print("band presence: ",band_present)
    print("band scores:",band_scores)

    debug = image.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for idx, (lane_box, has_band) in enumerate(zip(lane_boxes, band_present), 1):
        lx1, ly1, lx2, ly2 = lane_box
        color = (0, 255, 0) if has_band else (0, 0, 255)
        cv2.rectangle(debug, (lx1, ly1), (lx2, ly2), color, 2)
        label = f"{idx}: {'BAND' if has_band else 'WHITE'}"
        cv2.putText(debug, label, (lx1 + 2, ly1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    debug_path = args.debug_out or _default_debug_path(args.image)
    cv2.imwrite(debug_path, debug)


def test_run() -> None:
    """
    Test runner that processes all images in NeoBio Input Images folder.
    Detects blot bbox for each image and saves debug output.
    """
    input_dir = "NeoBio Input Images"
    
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"No images found in '{input_dir}'")
        return
    
    print(f"Found {len(image_files)} image(s) in '{input_dir}'")
    print("-" * 70)
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        print(f"[{idx}/{len(image_files)}] Processing: {image_file}")
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ✗ Failed to read image")
                continue
            
            result = detect_lane_bands(image, band_mode="mask")  # mask-only test run
            roi = result["roi"]
            lane_boxes = result["lane_boxes"]
            band_present = result["band_present"]

            print("band presence: ",band_present)

            debug = image.copy()
            x1, y1, x2, y2 = roi
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for idx_lane, (lane_box, has_band) in enumerate(zip(lane_boxes, band_present), 1):
                lx1, ly1, lx2, ly2 = lane_box
                color = (0, 255, 0) if has_band else (0, 0, 255)
                cv2.rectangle(debug, (lx1, ly1), (lx2, ly2), color, 2)
                label = f"{idx_lane}: {'BAND' if has_band else 'WHITE'}"
                cv2.putText(debug, label, (lx1 + 2, ly1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            debug_path = _default_debug_path(image_path)
            cv2.imwrite(debug_path, debug)
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()


if __name__ == "__main__":
    test_run()
    # main()