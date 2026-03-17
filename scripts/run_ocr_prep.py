#!/usr/bin/env python3
"""
Single-image OCR preparation runner.

Produces a stitched image of the regions above and below the blot ROI,
suitable for downstream OCR processing.  Does NOT perform OCR text extraction.

Usage
-----
    PYTHONPATH=src python scripts/run_ocr_prep.py path/to/image.jpg

Options
-------
    --top-extra-bottom-px INT   Pixels to extend top strip into the ROI
                                (default: 0)
    --bottom-extra-top-px INT   Pixels to extend bottom strip into the ROI
                                (default: 0)
    --gap-px INT                Separator height between stitched regions
                                (default: 20)
    --gap-value INT             Separator fill value 0-255, 255=white
                                (default: 255)
    --debug                     Save intermediate debug images
    --debug-dir PATH            Root directory for debug artefacts
                                (default: "ocr_prep_debug")
    --stitched-out PATH         Where to save the final stitched image.
                                Defaults to <debug-dir>/<stem>_stitched.png
    --print-json                Print result metadata as JSON to stdout
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neobio.pipelines import run_ocr_prep_pipeline
from neobio.utils import ensure_dir, default_out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare stitched OCR-input image from a blot image."
    )
    parser.add_argument("image", help="Path to input blot image")
    parser.add_argument(
        "--top-extra-bottom-px",
        type=int,
        default=0,
        metavar="INT",
        help="Pixels to extend top strip downward into the ROI (default: 0)",
    )
    parser.add_argument(
        "--bottom-extra-top-px",
        type=int,
        default=0,
        metavar="INT",
        help="Pixels to extend bottom strip upward into the ROI (default: 0)",
    )
    parser.add_argument(
        "--gap-px",
        type=int,
        default=20,
        metavar="INT",
        help="Separator height between stitched regions in pixels (default: 20)",
    )
    parser.add_argument(
        "--gap-value",
        type=int,
        default=255,
        metavar="INT",
        help="Separator fill value 0-255, 255=white (default: 255)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images",
    )
    parser.add_argument(
        "--debug-dir",
        default="testing output images/ocr_prep_debug",
        metavar="PATH",
        help="Root directory for debug artefacts (default: ocr_prep_debug)",
    )
    parser.add_argument(
        "--stitched-out",
        metavar="PATH",
        help="Explicit output path for the stitched image (optional)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print result metadata as JSON to stdout",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    try:
        result = run_ocr_prep_pipeline(
            image,
            top_extra_bottom_px=args.top_extra_bottom_px,
            bottom_extra_top_px=args.bottom_extra_top_px,
            gap_px=args.gap_px,
            gap_value=args.gap_value,
            debug=args.debug,
            debug_dir=args.debug_dir,
            input_path=args.image,
        )
    except Exception as exc:
        print(f"ERROR: Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save stitched image
    # ------------------------------------------------------------------
    if args.stitched_out:
        stitched_path = args.stitched_out
        ensure_dir(str(Path(stitched_path).parent))
    else:
        stitched_path = default_out_path(
            args.image, args.debug_dir, suffix="_stitched.png"
        )

    cv2.imwrite(stitched_path, result["stitched_image"])
    print(f"Stitched image saved: {stitched_path}")

    if args.debug and result["debug_dir"]:
        print(f"Debug artefacts saved: {result['debug_dir']}")

    # ------------------------------------------------------------------
    # Optional JSON metadata
    # ------------------------------------------------------------------
    if args.print_json:
        meta = {
            "roi": [int(v) for v in result["roi"]],
            "top_region_bounds": [int(v) for v in result["top_region_bounds"]],
            "bottom_region_bounds": [int(v) for v in result["bottom_region_bounds"]],
            "top_region_shape": list(result["top_region_shape"]),
            "bottom_region_shape": list(result["bottom_region_shape"]),
            "stitched_shape": list(result["stitched_shape"]),
            "stitched_path": stitched_path,
            "debug_dir": result["debug_dir"],
        }
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
